/**
 * Reliable text entry policy for the CUA bridge.
 *
 * The legacy Linux CUA server implements `type_text` with pynput. Pynput can
 * type ASCII directly, but arbitrary Unicode characters require temporary X11
 * keyboard-map entries. Long Unicode strings can therefore arrive in apps with
 * duplicated, missing, or reordered characters. Clipboard paste sends the
 * Unicode string as one value and avoids that per-character remapping.
 */

const TEXT_INPUT_MODES = new Set(["auto", "keystroke", "clipboard"]);
const NON_ASCII_RE = /[^\x00-\x7f]/u;

/** Parse and validate the developer-controlled input policy. */
export function parseTextInputMode(value) {
  const mode = (value || "auto").trim().toLowerCase();
  if (!TEXT_INPUT_MODES.has(mode)) {
    throw new Error(
      `Invalid CUA_TEXT_INPUT_MODE=${JSON.stringify(mode)}; ` +
      'expected "auto", "keystroke", or "clipboard".'
    );
  }
  return mode;
}

/** Return true when this call should use clipboard paste. */
export function shouldPasteText(text, mode = "auto") {
  if (mode === "clipboard") return true;
  if (mode === "keystroke") return false;
  return NON_ASCII_RE.test(text);
}

/** Resolve the platform paste shortcut used by the CUA server. */
export function pasteKeys(platform = process.platform) {
  return [platform === "darwin" ? "cmd" : "ctrl", "v"];
}

/**
 * Enter text without exposing the transport choice to the agent.
 *
 * In auto mode, ASCII keeps the existing type_text behavior while any Unicode
 * text is placed on the desktop clipboard and pasted into the focused field.
 * We intentionally leave the clipboard populated: restoring it immediately
 * can race with applications that consume the paste event asynchronously.
 */
export async function typeTextReliably(
  client,
  text,
  { mode = "auto", platform = process.platform } = {}
) {
  if (shouldPasteText(text, mode)) {
    await client.sendCommand("set_clipboard", { text });
    await client.sendCommand("hotkey", { keys: pasteKeys(platform) });
    return "clipboard";
  }

  await client.sendCommand("type_text", { text });
  return "keystroke";
}
