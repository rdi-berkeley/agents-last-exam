import assert from "node:assert/strict";
import test from "node:test";
import { parseTextInputMode, typeTextReliably } from "../src/text-input.js";

test("input policy validates configuration", () => {
  assert.equal(parseTextInputMode(undefined), "auto");
  assert.equal(parseTextInputMode(" Keystroke "), "keystroke");
  assert.throws(() => parseTextInputMode("invalid"), /CUA_TEXT_INPUT_MODE/);
});

test("Windows Unicode stays on the native server path without clipboard writes", async () => {
  const commands = [];
  const client = { async sendCommand(command, params) { commands.push({ command, params }); } };
  const text = "\u4e2d\u6587 العربية 한글 e\u0301 \ud83d\ude42";
  assert.equal(await typeTextReliably(client, text, { platform: "win32" }), "keystroke");
  assert.deepEqual(commands, [{ command: "type_text", params: { text } }]);
});

test("Linux ASCII and explicit keystrokes retain the server input path", async () => {
  const calls = [];
  const client = { async sendCommand(command, params) { calls.push({ command, params }); } };
  await typeTextReliably(client, "ASCII", { platform: "linux" });
  await typeTextReliably(client, "\u4e2d", { mode: "keystroke", platform: "linux" });
  assert.deepEqual(calls.map(call => call.command), ["type_text", "type_text"]);
});

test("the next input cannot overwrite data before delayed consumption", async () => {
  const calls = [];
  let consume;
  const client = {
    async sendCommand(command, params) {
      calls.push({ command, params });
      if (command === "run_command") {
        await new Promise(resolve => { consume = resolve; });
        return { return_code: 0, stdout: JSON.stringify({ consumed: true, bytes: 3 }) };
      }
    },
  };
  const first = typeTextReliably(client, "\u4e2d", { platform: "linux" });
  const second = typeTextReliably(client, "ASCII", { platform: "linux" });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls.map(call => call.command), ["run_command"]);
  assert.equal(calls[0].params.timeout, 20);
  consume();
  assert.deepEqual(await Promise.all([first, second]), ["clipboard", "keystroke"]);
  assert.deepEqual(calls.map(call => call.command), ["run_command", "type_text"]);
});

for (const result of [
  { return_code: 1, stderr: "not consumed" },
  { return_code: 0, stdout: '{"consumed":false,"bytes":3}' },
  { return_code: 0, stdout: '{"consumed":true,"bytes":99}' },
]) {
  test(`paste errors surface without retrying input: ${JSON.stringify(result)}`, async () => {
    const calls = [];
    const client = { async sendCommand(command) { calls.push(command); return result; } };
    const failed = typeTextReliably(client, "\u4e2d", { platform: "linux" });
    const next = typeTextReliably(client, "ASCII", { platform: "linux" });
    await assert.rejects(failed);
    assert.equal(await next, "keystroke");
    assert.deepEqual(calls, ["run_command", "type_text"]);
  });
}

test("empty input does not wait for a nonexistent clipboard consumer", async () => {
  const client = { sendCommand() { assert.fail("unexpected command"); } };
  assert.equal(await typeTextReliably(client, "", { platform: "linux", mode: "clipboard" }), "empty");
});

test("independent desktops have independent queues", async () => {
  let consume;
  const busy = { sendCommand: () => new Promise(resolve => { consume = resolve; }) };
  const free = { async sendCommand() {} };
  const pending = typeTextReliably(busy, "busy");
  assert.equal(await typeTextReliably(free, "free"), "keystroke");
  consume();
  await pending;
});

test("clipboard-only mode never silently uses an unsafe non-X11 paste", async () => {
  const client = { sendCommand() { assert.fail("unexpected command"); } };
  await assert.rejects(typeTextReliably(client, "text", { mode: "clipboard", platform: "win32" }), /requires X11/);
});

test("large Unicode text uses a persistent payload file rather than shell arguments", async () => {
  const text = "\u4e2d".repeat(30000);
  const calls = [];
  const client = {
    async sendCommand(command, params) {
      calls.push({ command, params });
      if (command === "run_command" && params.timeout === 5) {
        return { return_code: 0, stdout: '"/home/user/.cache/ale-cua/input.txt"' };
      }
      if (command === "run_command") {
        assert.ok(params.command.startsWith("exec "));
        assert.match(params.command, /--file/);
        assert.ok(params.command.length < 20000);
        return { return_code: 0, stdout: JSON.stringify({ consumed: true, bytes: Buffer.byteLength(text) }) };
      }
    },
  };
  await typeTextReliably(client, text, { platform: "linux" });
  assert.deepEqual(calls.map(call => call.command), ["run_command", "write_bytes", "run_command", "delete_file"]);
  assert.equal(Buffer.from(calls[1].params.content_b64, "base64").toString(), text);
});
