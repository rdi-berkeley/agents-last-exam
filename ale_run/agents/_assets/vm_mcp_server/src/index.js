#!/usr/bin/env node
/**
 * VM MCP Server — non-GUI VM primitives over MCP (stdio).
 *
 * A thin substrate: it exposes the CUA computer-server's native, irreducible
 * non-GUI capabilities as MCP tools, and nothing more. Each maps to one native
 * capability, none substitutable by another:
 *   - process execution     → run_command (one-shot)
 *   - interactive execution → pty_* (streaming, stateful PTY sessions)
 *   - filesystem I/O         → read_text / write_text / read_bytes / write_bytes
 *   - clipboard              → read_clipboard / write_clipboard
 *
 * Deliberately NOT included: convenience ops reducible to run_command
 * (ls/test/stat/mkdir/rm/rmdir), and opinionated/composite tools (edit, glob,
 * grep, a "bash" alias). Agents compose those themselves on top of these
 * primitives — this server does not make tool-design decisions for them. The
 * GUI action set (mouse/keyboard/screenshot) is the sibling cua_mcp_server.
 *
 * Reads CUA_SERVER_URL (injected by the executor via cua_bridge_env) and falls
 * back to http://localhost:5000. Runs in the substrate, consumed by the agent
 * CLI over stdio.
 *
 * Usage:
 *   node src/index.js                       # start MCP server (stdio)
 *   node src/index.js --test                # smoke test against the CUA server
 *   CUA_SERVER_URL=http://...:5000 node src/index.js
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { VmClient } from "./vm-client.js";

const CUA_URL = process.env.CUA_SERVER_URL || "http://localhost:5000";
const client = new VmClient(CUA_URL);

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function textOnly(text) {
  return { content: [{ type: "text", text }] };
}

function errorResult(text) {
  return { content: [{ type: "text", text }], isError: true };
}

// ------------------------------------------------------------------
// PTY session registry (bridge-side buffering)
//
// The computer-server only delivers PTY output to subscribers connected
// at the time it is produced — there is no backlog. So we open a
// persistent SSE subscription at pty_start and accumulate output into a
// per-session buffer that pty_read drains. This presents a poll-style
// read over the server's push-style stream.
// ------------------------------------------------------------------

const MAX_PTY_BUFFER = 2_000_000; // bytes retained per session before head-dropping

/** @type {Map<number, {chunks: Buffer[], bytes: number, exited: boolean, exitCode: (number|null), truncated: boolean, waiters: Array<() => void>, abort: AbortController}>} */
const ptys = new Map();

function ptyNotify(session) {
  const waiters = session.waiters;
  session.waiters = [];
  for (const resolve of waiters) resolve();
}

function ptyPush(session, buf) {
  if (buf.length === 0) return;
  session.chunks.push(buf);
  session.bytes += buf.length;
  while (session.bytes > MAX_PTY_BUFFER && session.chunks.length > 1) {
    const dropped = session.chunks.shift();
    session.bytes -= dropped.length;
    session.truncated = true;
  }
  ptyNotify(session);
}

function ptyDrain(session) {
  const out = Buffer.concat(session.chunks).toString("utf8");
  session.chunks = [];
  session.bytes = 0;
  const truncated = session.truncated;
  session.truncated = false;
  return { out, truncated };
}

/** Wait up to timeoutMs for new output or exit; resolves early on either. */
function ptyWait(session, timeoutMs) {
  if (session.chunks.length > 0 || session.exited) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      // drop this waiter so it is not resolved later
      session.waiters = session.waiters.filter((w) => w !== wrapped);
      resolve();
    }, timeoutMs);
    const wrapped = () => {
      clearTimeout(timer);
      resolve();
    };
    session.waiters.push(wrapped);
  });
}

// ------------------------------------------------------------------
// --test mode
// ------------------------------------------------------------------
if (process.argv.includes("--test")) {
  console.log(`Testing VM MCP Server against ${CUA_URL} ...`);
  try {
    const r = await client.sendCommand("run_command", { command: "echo vm-bridge-ok" });
    console.log(`  run_command: ${JSON.stringify(r.stdout)} rc=${r.return_code}`);

    const tmp = "/tmp/vm_bridge_test.txt";
    await client.sendCommand("write_text", { path: tmp, content: "hello-vm" });
    const rd = await client.sendCommand("read_text", { path: tmp });
    console.log(`  write_text/read_text: ${JSON.stringify(rd.content)}`);
    const rb = await client.sendCommand("read_bytes", { path: tmp });
    console.log(`  read_bytes: ${(rb.content_b64 || "").length} chars b64`);
    await client.sendCommand("run_command", { command: `rm -f '${tmp}'` });

    await client.sendCommand("set_clipboard", { text: "clip-roundtrip-ok" });
    const clip = await client.sendCommand("copy_to_clipboard");
    console.log(`  clipboard write/read: ${JSON.stringify(clip.content)}`);

    // PTY round-trip
    const info = await client.ptyCreate({ command: "bash", cols: 80, rows: 24 });
    const pid = info.pid;
    console.log(`  pty_start: pid=${pid}`);
    const chunks = [];
    let exited = false;
    const ac = new AbortController();
    const streaming = client.ptyStream(
      pid,
      { onOutput: (b) => chunks.push(b), onExit: () => { exited = true; } },
      ac.signal,
    );
    await new Promise((res) => setTimeout(res, 300)); // let subscription attach
    await client.ptyStdin(pid, Buffer.from("echo pty-roundtrip-ok\n"));
    await new Promise((res) => setTimeout(res, 800));
    console.log(`  pty output: ${JSON.stringify(Buffer.concat(chunks).toString("utf8").trim())}`);
    await client.ptyKill(pid);
    ac.abort();
    await streaming.catch(() => {});
    console.log(`  pty_kill: ok (exited=${exited})`);

    console.log("All tests passed.");
    process.exit(0);
  } catch (e) {
    console.error(`Test failed: ${e.message}`);
    process.exit(1);
  }
}

// ------------------------------------------------------------------
// MCP Server
// ------------------------------------------------------------------
const server = new McpServer({
  name: "vm-primitives",
  version: "0.1.0",
});

// ================================================================
// Process execution
// ================================================================

server.registerTool(
  "run_command",
  {
    description:
      "Run a command on the VM and wait for it to finish. One-shot and blocking: it does not stream and is " +
      "not interactive, and no shell state persists between calls (set cwd inline with `cd /path && ...`, " +
      "pass env inline with `VAR=val cmd`). For long-running, interactive, or streaming processes use the " +
      "pty_* tools. Returns stdout, stderr, and the exit code.",
    inputSchema: {
      command: z.string().describe("The command to run (executed via the VM's system shell)."),
      timeout: z
        .number()
        .optional()
        .describe("Optional server-side timeout in seconds; the command is terminated if it runs longer."),
    },
    // Structured output: the three fields travel as data, not flattened prose.
    // Reverse-parsing the text block is lossy (stdout may itself contain a line
    // "stderr:", and the stderr section is conditionally present), so machine
    // consumers read structuredContent; the text block stays for readability.
    outputSchema: {
      exit_code: z.number().int().describe("Process exit code."),
      stdout: z.string().describe("Captured standard output."),
      stderr: z.string().describe("Captured standard error."),
    },
  },
  async ({ command, timeout }) => {
    const params = { command };
    if (timeout !== undefined) params.timeout = timeout;
    // Generous client-side headroom over the server-side timeout.
    const clientTimeout = timeout !== undefined ? (timeout + 30) * 1000 : 120000;
    const r = await client.sendCommand("run_command", params, clientTimeout);
    const structured = {
      exit_code: r.return_code ?? 0,
      stdout: r.stdout ?? "",
      stderr: r.stderr ?? "",
    };
    const parts = [`exit_code: ${structured.exit_code}`, `stdout:\n${structured.stdout}`];
    if (structured.stderr) parts.push(`stderr:\n${structured.stderr}`);
    return {
      content: [{ type: "text", text: parts.join("\n") }],
      structuredContent: structured,
    };
  },
);

// ================================================================
// Filesystem I/O
// ================================================================

server.tool(
  "read_text",
  "Read a UTF-8 text file from the VM and return its full contents. Use read_bytes for binary files or to " +
    "read a byte range.",
  {
    path: z.string().describe("Absolute path on the VM (a leading ~ is expanded to the user's home)."),
  },
  async ({ path }) => {
    const r = await client.sendCommand("read_text", { path });
    return textOnly(r.content ?? "");
  },
);

server.tool(
  "write_text",
  "Write UTF-8 text to a file on the VM, overwriting any existing contents (creates the file if absent). " +
    "Use write_bytes for binary data or to append.",
  {
    path: z.string().describe("Absolute path on the VM (a leading ~ is expanded to the user's home)."),
    content: z.string().describe("The text content to write."),
  },
  async ({ path, content }) => {
    await client.sendCommand("write_text", { path, content });
    return textOnly(`Wrote ${Buffer.byteLength(content, "utf8")} bytes to ${path}`);
  },
);

server.tool(
  "read_bytes",
  "Read raw bytes from a file on the VM and return them base64-encoded. Supports a byte range via " +
    "offset/length.",
  {
    path: z.string().describe("Absolute path on the VM."),
    offset: z.number().int().min(0).optional().describe("Byte offset to start reading from (default 0)."),
    length: z.number().int().min(0).optional().describe("Number of bytes to read (default: to end of file)."),
  },
  async ({ path, offset, length }) => {
    const params = { path };
    if (offset !== undefined) params.offset = offset;
    if (length !== undefined) params.length = length;
    const r = await client.sendCommand("read_bytes", params);
    return textOnly(r.content_b64 ?? "");
  },
);

server.tool(
  "write_bytes",
  "Write raw bytes (provided base64-encoded) to a file on the VM. Can append instead of overwriting.",
  {
    path: z.string().describe("Absolute path on the VM."),
    content_b64: z.string().describe("Base64-encoded bytes to write."),
    append: z.boolean().optional().describe("If true, append to the file instead of overwriting (default false)."),
  },
  async ({ path, content_b64, append }) => {
    const params = { path, content_b64 };
    if (append !== undefined) params.append = append;
    await client.sendCommand("write_bytes", params);
    return textOnly(`Wrote bytes to ${path}${append ? " (appended)" : ""}`);
  },
);

// ================================================================
// Clipboard
//
// The VM's system clipboard is a distinct capability — it is not reachable
// through run_command (it needs the desktop session). Tool names are clarified
// here because the native command names are misleading: the server's
// `copy_to_clipboard` actually READS the clipboard and `set_clipboard` writes.
// ================================================================

server.tool(
  "read_clipboard",
  "Read the current text contents of the VM's system clipboard. (Maps to the server's copy_to_clipboard.)",
  {},
  async () => {
    const r = await client.sendCommand("copy_to_clipboard");
    return textOnly(r.content ?? "");
  },
);

server.tool(
  "write_clipboard",
  "Set the VM's system clipboard to the given text. (Maps to the server's set_clipboard.)",
  {
    text: z.string().describe("Text to place on the clipboard."),
  },
  async ({ text }) => {
    await client.sendCommand("set_clipboard", { text });
    return textOnly(`Set clipboard (${Buffer.byteLength(text, "utf8")} bytes)`);
  },
);

// ================================================================
// Interactive execution (PTY)
// ================================================================

server.tool(
  "pty_start",
  "Start a PTY (pseudo-terminal) session on the VM for a long-running or interactive process. " +
    "Returns a pid used by pty_input / pty_read / pty_resize / pty_kill. Unlike run_command, a PTY persists " +
    "across calls and streams output incrementally — read it with pty_read. Defaults to an interactive " +
    "bash shell if no command is given.",
  {
    command: z
      .string()
      .optional()
      .describe("Program to launch in the PTY (default: an interactive `bash` shell)."),
    cwd: z.string().optional().describe("Working directory for the process."),
    env: z.record(z.string()).optional().describe("Extra environment variables for the process."),
    cols: z.number().int().positive().optional().describe("Terminal width in columns (default 80)."),
    rows: z.number().int().positive().optional().describe("Terminal height in rows (default 24)."),
  },
  async ({ command, cwd, env, cols, rows }) => {
    const opts = {};
    if (command !== undefined) opts.command = command;
    if (cwd !== undefined) opts.cwd = cwd;
    if (env !== undefined) opts.envs = env;
    if (cols !== undefined) opts.cols = cols;
    if (rows !== undefined) opts.rows = rows;

    const info = await client.ptyCreate(opts);
    const pid = info.pid;
    const abort = new AbortController();
    const session = {
      chunks: [],
      bytes: 0,
      exited: false,
      exitCode: null,
      truncated: false,
      waiters: [],
      abort,
    };
    ptys.set(pid, session);

    // Open the persistent subscription immediately; output produced before
    // this attaches (e.g. the initial shell prompt) may not be captured.
    client
      .ptyStream(
        pid,
        {
          onOutput: (buf) => ptyPush(session, buf),
          onExit: (code) => {
            session.exited = true;
            session.exitCode = code;
            ptyNotify(session);
          },
        },
        abort.signal,
      )
      .catch(() => {
        // Stream errored/ended; mark exited so readers don't block forever.
        session.exited = true;
        ptyNotify(session);
      });

    return textOnly(`pid: ${pid}\ncols: ${info.cols}\nrows: ${info.rows}`);
  },
);

server.tool(
  "pty_input",
  "Write raw bytes to a running PTY session's stdin. Include a trailing newline in `data` to submit a " +
    "command; control characters are allowed (e.g. \\u0003 for Ctrl-C). Call pty_read to see the output.",
  {
    pid: z.number().int().describe("PTY session id returned by pty_start."),
    data: z.string().describe("Raw text/bytes to write to stdin (UTF-8). Include \\n to submit a line."),
  },
  async ({ pid, data }) => {
    if (!ptys.has(pid)) return errorResult(`No PTY session with pid ${pid}`);
    await client.ptyStdin(pid, Buffer.from(data, "utf8"));
    return textOnly(`Sent ${Buffer.byteLength(data, "utf8")} bytes to pid ${pid}`);
  },
);

server.tool(
  "pty_read",
  "Read buffered output from a PTY session since the last read. If no output is available yet, waits up to " +
    "timeout_ms for some (or for the process to exit). Output is raw terminal bytes and may contain ANSI " +
    "escape sequences. Reports whether the process is still running or has exited.",
  {
    pid: z.number().int().describe("PTY session id returned by pty_start."),
    timeout_ms: z
      .number()
      .int()
      .min(0)
      .optional()
      .describe("Max time to wait for output if none is buffered (default 2000ms; 0 returns immediately)."),
  },
  async ({ pid, timeout_ms }) => {
    const session = ptys.get(pid);
    if (!session) return errorResult(`No PTY session with pid ${pid}`);

    await ptyWait(session, timeout_ms ?? 2000);
    const { out, truncated } = ptyDrain(session);

    const status = session.exited
      ? `[exited, code ${session.exitCode}]`
      : "[running]";
    const head = truncated ? "[earlier output dropped — buffer overflow]\n" : "";
    if (session.exited && out.length === 0) {
      ptys.delete(pid);
    }
    return textOnly(`${head}${out}${out.endsWith("\n") || out.length === 0 ? "" : "\n"}${status}`);
  },
);

server.tool(
  "pty_resize",
  "Resize a PTY session's terminal dimensions.",
  {
    pid: z.number().int().describe("PTY session id returned by pty_start."),
    cols: z.number().int().positive().describe("New terminal width in columns."),
    rows: z.number().int().positive().describe("New terminal height in rows."),
  },
  async ({ pid, cols, rows }) => {
    if (!ptys.has(pid)) return errorResult(`No PTY session with pid ${pid}`);
    await client.ptyResize(pid, cols, rows);
    return textOnly(`Resized pid ${pid} to ${cols}x${rows}`);
  },
);

server.tool(
  "pty_kill",
  "Terminate a PTY session and release its resources.",
  {
    pid: z.number().int().describe("PTY session id returned by pty_start."),
  },
  async ({ pid }) => {
    const session = ptys.get(pid);
    if (!session) return errorResult(`No PTY session with pid ${pid}`);
    try {
      await client.ptyKill(pid);
    } finally {
      session.abort.abort();
      ptys.delete(pid);
    }
    return textOnly(`Killed pid ${pid}`);
  },
);

// ------------------------------------------------------------------
// Start stdio transport
// ------------------------------------------------------------------
const transport = new StdioServerTransport();
await server.connect(transport);
