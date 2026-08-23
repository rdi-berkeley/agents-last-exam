import assert from "node:assert/strict";
import test from "node:test";

import {
  parseTextInputMode,
  pasteKeys,
  shouldPasteText,
  typeTextReliably,
} from "../src/text-input.js";

function recordingClient() {
  const calls = [];
  return {
    calls,
    async sendCommand(command, params) {
      calls.push({ command, params });
      return { success: true };
    },
  };
}

test("text input mode defaults to auto and is case-insensitive", () => {
  assert.equal(parseTextInputMode(undefined), "auto");
  assert.equal(parseTextInputMode(" Clipboard "), "clipboard");
});

test("text input mode rejects unknown values", () => {
  assert.throws(
    () => parseTextInputMode("unicode"),
    /Invalid CUA_TEXT_INPUT_MODE/
  );
});

test("auto mode uses keystrokes for ASCII and clipboard for Unicode", () => {
  assert.equal(shouldPasteText("user@example.com", "auto"), false);
  assert.equal(shouldPasteText("Valuation: $700M\n", "auto"), false);
  assert.equal(shouldPasteText("估值为7亿美元", "auto"), true);
  assert.equal(shouldPasteText("hello🙂", "auto"), true);
  assert.equal(shouldPasteText("full-width，comma", "auto"), true);
});

test("explicit modes override text inspection", () => {
  assert.equal(shouldPasteText("中文", "keystroke"), false);
  assert.equal(shouldPasteText("ASCII", "clipboard"), true);
});

test("paste shortcut is platform-aware", () => {
  assert.deepEqual(pasteKeys("linux"), ["ctrl", "v"]);
  assert.deepEqual(pasteKeys("win32"), ["ctrl", "v"]);
  assert.deepEqual(pasteKeys("darwin"), ["cmd", "v"]);
});

test("Unicode is copied and pasted without calling type_text", async () => {
  const client = recordingClient();
  const text = "中文ABC，🙂ALE2.0真棒！";

  const route = await typeTextReliably(client, text, {
    mode: "auto",
    platform: "linux",
  });

  assert.equal(route, "clipboard");
  assert.deepEqual(client.calls, [
    { command: "set_clipboard", params: { text } },
    { command: "hotkey", params: { keys: ["ctrl", "v"] } },
  ]);
});

test("ASCII preserves the existing type_text path", async () => {
  const client = recordingClient();
  const text = "Valuation: $700M";

  const route = await typeTextReliably(client, text, { mode: "auto" });

  assert.equal(route, "keystroke");
  assert.deepEqual(client.calls, [
    { command: "type_text", params: { text } },
  ]);
});

test("clipboard failure is surfaced and does not fall back to keystrokes", async () => {
  const calls = [];
  const client = {
    async sendCommand(command, params) {
      calls.push({ command, params });
      throw new Error("clipboard unavailable");
    },
  };

  await assert.rejects(
    typeTextReliably(client, "中文", { mode: "auto" }),
    /clipboard unavailable/
  );
  assert.deepEqual(calls, [
    { command: "set_clipboard", params: { text: "中文" } },
  ]);
});
