/**
 * HTTP client for the CUA computer-server, covering the non-GUI surface:
 *   - one-shot commands via POST /cmd (SSE-framed JSON response)
 *   - PTY sessions via the /pty* REST + SSE-stream endpoints
 *
 * The /cmd transport mirrors the GUI bridge's cua-client.js. PTY support is
 * added here because PTY lives on dedicated HTTP routes, not /cmd.
 */

export class VmClient {
  /**
   * @param {string} serverUrl - CUA server base URL (default: http://localhost:5000)
   * @param {number} timeout - Request timeout in ms for one-shot commands (default: 30000)
   */
  constructor(serverUrl = "http://localhost:5000", timeout = 30000) {
    this.serverUrl = serverUrl.replace(/\/+$/, "");
    this.timeout = timeout;
  }

  /**
   * Send a one-shot command to the CUA server and return the parsed result.
   * The server frames its JSON reply as a single SSE `data:` line.
   * @param {string} command
   * @param {Record<string, unknown>} params
   * @param {number} [timeoutMs] - per-call timeout override (ms)
   * @returns {Promise<Record<string, unknown>>}
   */
  async sendCommand(command, params = {}, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs ?? this.timeout);

    try {
      const resp = await fetch(`${this.serverUrl}/cmd`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command, params }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`CUA server HTTP ${resp.status}: ${text}`);
      }

      const body = await resp.text();
      let result = null;
      for (const line of body.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            result = JSON.parse(line.slice(6));
          } catch {
            // skip malformed lines
          }
        }
      }

      if (!result) {
        throw new Error(`No valid response for command '${command}'`);
      }
      if (result.success === false) {
        throw new Error(`Command '${command}' failed: ${result.error ?? "unknown error"}`);
      }
      return result;
    } finally {
      clearTimeout(timer);
    }
  }

  // ------------------------------------------------------------------
  // PTY session endpoints
  // ------------------------------------------------------------------

  /**
   * Create a PTY session.
   * @param {{command?: string, cols?: number, rows?: number, cwd?: string, envs?: Record<string,string>}} opts
   * @returns {Promise<{pid: number, cols: number, rows: number}>}
   */
  async ptyCreate(opts = {}) {
    const resp = await fetch(`${this.serverUrl}/pty`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`PTY create HTTP ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  /**
   * Write raw bytes to a PTY's stdin.
   * @param {number} pid
   * @param {Buffer} data
   */
  async ptyStdin(pid, data) {
    const resp = await fetch(`${this.serverUrl}/pty/${pid}/stdin`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: data.toString("base64") }),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`PTY stdin HTTP ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  /** Resize a PTY. */
  async ptyResize(pid, cols, rows) {
    const resp = await fetch(`${this.serverUrl}/pty/${pid}/resize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cols, rows }),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`PTY resize HTTP ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  /** Kill a PTY session. */
  async ptyKill(pid) {
    const resp = await fetch(`${this.serverUrl}/pty/${pid}`, { method: "DELETE" });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`PTY kill HTTP ${resp.status}: ${text}`);
    }
    return resp.json();
  }

  /**
   * Subscribe to a PTY's SSE output stream. Invokes callbacks as events
   * arrive and resolves when the stream ends (process exit or abort).
   *
   * IMPORTANT: call this immediately after ptyCreate — output is only
   * delivered to subscribers connected at the time it is produced; the
   * server keeps no backlog for late subscribers.
   *
   * @param {number} pid
   * @param {{onOutput: (buf: Buffer) => void, onExit: (code: number) => void}} cbs
   * @param {AbortSignal} signal
   */
  async ptyStream(pid, { onOutput, onExit }, signal) {
    const resp = await fetch(`${this.serverUrl}/pty/${pid}/stream`, {
      headers: { Accept: "text/event-stream" },
      signal,
    });
    if (!resp.ok || !resp.body) {
      const text = await resp.text().catch(() => "");
      throw new Error(`PTY stream HTTP ${resp.status}: ${text}`);
    }

    const decoder = new TextDecoder();
    let buf = "";
    try {
      for await (const chunk of resp.body) {
        buf += decoder.decode(chunk, { stream: true });
        let idx;
        // SSE events are separated by a blank line.
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const event = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const line of event.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            let payload;
            try {
              payload = JSON.parse(line.slice(6));
            } catch {
              continue;
            }
            if (payload.type === "output") {
              onOutput(Buffer.from(payload.data ?? "", "base64"));
            } else if (payload.type === "exit") {
              onExit(payload.code ?? 0);
              return;
            }
          }
        }
      }
    } catch (e) {
      if (e?.name === "AbortError") return;
      throw e;
    }
  }
}
