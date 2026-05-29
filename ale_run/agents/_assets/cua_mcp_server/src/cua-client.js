/**
 * HTTP client for CUA computer-server.
 *
 * Sends commands via POST /cmd and parses SSE-streamed responses.
 * Aligned with the OpenClaw cua-plugin client at
 * agenthle/orchestration/external/openclaw/openclaw/cua-plugin/src/cua-client.ts.
 */

export class CuaClient {
  /**
   * @param {string} serverUrl - CUA server base URL (default: http://localhost:5000)
   * @param {number} timeout - Request timeout in ms (default: 30000)
   */
  constructor(serverUrl = "http://localhost:5000", timeout = 30000) {
    this.serverUrl = serverUrl.replace(/\/+$/, "");
    this.timeout = timeout;
  }

  /**
   * Send a command to the CUA server and return the parsed result.
   * @param {string} command
   * @param {Record<string, unknown>} params
   * @returns {Promise<Record<string, unknown>>}
   */
  async sendCommand(command, params = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

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
        throw new Error(
          `Command '${command}' failed: ${result.error ?? "unknown error"}`
        );
      }

      return result;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Take a screenshot and return raw base64 image data.
   * @returns {Promise<{base64: string, mimeType: string}>}
   */
  async screenshot() {
    const result = await this.sendCommand("screenshot");
    const imageData = result.image_data;
    if (!imageData) {
      throw new Error("Screenshot returned no image data");
    }
    return { base64: imageData, mimeType: "image/png" };
  }

  /**
   * Get screen dimensions.
   * @returns {Promise<{width: number, height: number}>}
   */
  async getScreenSize() {
    const result = await this.sendCommand("get_screen_size");
    const size = result.size;
    if (size) return size;
    return { width: result.width, height: result.height };
  }
}
