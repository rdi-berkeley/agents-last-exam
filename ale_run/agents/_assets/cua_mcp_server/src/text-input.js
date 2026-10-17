import { readFileSync } from "node:fs";

const inputQueues = new WeakMap();
const helper = readFileSync(new URL("./x11-paste.py", import.meta.url)).toString("base64");

export function parseTextInputMode(value) {
  const mode = (value || "auto").trim().toLowerCase();
  if (!["auto", "keystroke", "clipboard"].includes(mode)) {
    throw new Error(`Invalid CUA_TEXT_INPUT_MODE=${JSON.stringify(mode)}; expected "auto", "keystroke", or "clipboard".`);
  }
  return mode;
}

export function typeTextReliably(client, text, {
  mode = "auto", platform = process.platform,
  python = process.env.CUA_INPUT_PYTHON || "python3",
} = {}) {
  mode = parseTextInputMode(mode);
  const previous = inputQueues.get(client) || Promise.resolve();
  const pending = previous.then(async () => {
    if (!text) return "empty";
    if (mode === "clipboard" && platform !== "linux") {
      throw new Error("Acknowledged clipboard input requires X11; use auto for native Unicode input on this platform.");
    }
    if (platform !== "linux" || mode === "keystroke" || (mode === "auto" && !/[^\x00-\x7f]/u.test(text))) {
      await client.sendCommand("type_text", { text });
      return "keystroke";
    }
    const payload = Buffer.from(text).toString("base64");
    const executable = `'${python.replaceAll("'", "'\\''")}'`;
    let payloadPath;
    try {
      let argument = `'${payload}'`;
      if (payload.length > 48000) {
        const staging = "import json,os,tempfile;from pathlib import Path;" +
          "p=Path.home()/'.cache'/'ale-cua';p.mkdir(parents=True,exist_ok=True,mode=0o700);" +
          "fd,name=tempfile.mkstemp(dir=p,suffix='.txt');os.close(fd);print(json.dumps(name))";
        const staged = await client.sendCommand("run_command", {
          command: `exec ${executable} -c "${staging}"`, timeout: 5,
        });
        if (staged.return_code !== 0) throw new Error(`Cannot stage input: ${staged.stderr}`);
        payloadPath = JSON.parse(staged.stdout.trim());
        await client.sendCommand("write_bytes", { path: payloadPath, content_b64: payload });
        argument = `--file '${payloadPath.replaceAll("'", "'\\''")}'`;
      }
      const result = await client.sendCommand("run_command", {
        command: `exec ${executable} -c "import base64;exec(base64.b64decode('${helper}'))" ${argument}`,
        timeout: 20,
      });
      if (result.return_code !== 0) {
        throw new Error(`Acknowledged X11 paste failed: ${result.stderr || result.stdout || result.return_code}`);
      }
      const receipt = JSON.parse(result.stdout.trim());
      if (receipt.consumed !== true || receipt.bytes !== Buffer.byteLength(text)) {
        throw new Error("X11 paste returned no matching consumption receipt");
      }
    } finally {
      if (payloadPath) await client.sendCommand("delete_file", { path: payloadPath }).catch(() => {});
    }
    return "clipboard";
  });
  inputQueues.set(client, pending.catch(() => {}));
  return pending;
}
