#!/usr/bin/env node
/**
 * CUA MCP Server — LiteDesktopActionSpace over MCP (stdio).
 *
 * Wraps CUA computer-server HTTP API into MCP tools aligned with
 * LiteDesktopActionSpace (normalized [0, 1000] coordinates).
 *
 * The MCP bridge converts normalized coordinates to absolute pixels
 * by querying screen size from the CUA server on first use, then caching.
 *
 * Runs on the VM, consumed by Claude Code / Codex via stdio transport.
 *
 * Usage:
 *   node src/index.js                     # start MCP server (stdio)
 *   node src/index.js --test              # run smoke test against CUA server
 *   CUA_SERVER_URL=http://...:5000 node src/index.js
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { CuaClient } from "./cua-client.js";

// Key normalization — maps LLM-style key names to CUA server pynput names.
// CUA server's Windows handler has its own _key_from_string with .lower(),
// but Linux handler uses raw getattr(Key, key) which requires exact pynput
// attribute names. Normalize here so all backends work uniformly.
const KEY_MAP = {
  // Arrow keys (browser-style → pynput)
  ARROWUP: "up",
  ARROWDOWN: "down",
  ARROWLEFT: "left",
  ARROWRIGHT: "right",
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
  // Modifier aliases → pynput attribute names
  control: "ctrl",
  Control: "ctrl",
  CONTROL: "ctrl",
  Ctrl: "ctrl",
  CTRL: "ctrl",
  Shift: "shift",
  SHIFT: "shift",
  Alt: "alt",
  ALT: "alt",
  option: "alt",
  Option: "alt",
  meta: "cmd",
  Meta: "cmd",
  command: "cmd",
  Command: "cmd",
  win: "cmd",
  Win: "cmd",
  super: "cmd",
  Super: "cmd",
  // Common key aliases
  Enter: "enter",
  ENTER: "enter",
  Return: "enter",
  return: "enter",
  Escape: "esc",
  escape: "esc",
  ESC: "esc",
  Space: "space",
  SPACE: "space",
  Tab: "tab",
  TAB: "tab",
  Backspace: "backspace",
  BACKSPACE: "backspace",
  Delete: "delete",
  DELETE: "delete",
  Home: "home",
  End: "end",
  PageUp: "page_up",
  pageup: "page_up",
  PageDown: "page_down",
  pagedown: "page_down",
  CapsLock: "caps_lock",
  capslock: "caps_lock",
  Insert: "insert",
  PrintScreen: "print_screen",
};

function normalizeKey(key) {
  return KEY_MAP[key] ?? key.toLowerCase();
}

const CUA_URL = process.env.CUA_SERVER_URL || "http://localhost:5000";
const client = new CuaClient(CUA_URL);

// ------------------------------------------------------------------
// Coordinate conversion: normalized [0, 1000] → absolute pixels
// ------------------------------------------------------------------

const COORD_MAX = 1000;
let _screenSize = null;

async function getScreenSize() {
  if (!_screenSize) {
    _screenSize = await client.getScreenSize();
  }
  return _screenSize;
}

async function toAbsolute(coordinate) {
  const screen = await getScreenSize();
  return {
    x: Math.round((coordinate[0] / COORD_MAX) * screen.width),
    y: Math.round((coordinate[1] / COORD_MAX) * screen.height),
  };
}

async function toNormalized(absX, absY) {
  const screen = await getScreenSize();
  return [
    Math.round((absX / screen.width) * COORD_MAX),
    Math.round((absY / screen.height) * COORD_MAX),
  ];
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function textOnly(label) {
  return { content: [{ type: "text", text: label }] };
}

// ------------------------------------------------------------------
// --test mode
// ------------------------------------------------------------------
if (process.argv.includes("--test")) {
  console.log(`Testing CUA MCP Server against ${CUA_URL} ...`);
  try {
    const size = await client.getScreenSize();
    console.log(`  Screen size: ${size.width}x${size.height}`);
    const shot = await client.screenshot();
    console.log(`  Screenshot: ${shot.base64.length} chars base64`);
    const abs = await toAbsolute([500, 500]);
    console.log(`  Coordinate [500, 500] -> pixel (${abs.x}, ${abs.y})`);
    await client.sendCommand("move_cursor", { x: abs.x, y: abs.y });
    console.log("  move_cursor: OK");
    const pos = await client.sendCommand("get_cursor_position");
    console.log(`  get_cursor_position: ${JSON.stringify(pos.position)}`);
    await client.sendCommand("key_down", { key: "shift" });
    await client.sendCommand("key_up", { key: "shift" });
    console.log("  key_down/key_up: OK");
    await client.sendCommand("scroll_direction", { direction: "down", clicks: 1 });
    console.log("  scroll_direction: OK");
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
  name: "cua-desktop",
  version: "0.3.0",
});

// Zod schema for normalized [0, 1000] coordinate pair
const zCoordinate = z.array(z.number()).length(2).describe(
  "(x, y) coordinates normalized to [0, 1000]."
);

// ================================================================
// Keyboard Actions
// ================================================================

server.tool(
  "key",
  "On a desktop, press and release keys. For hotkeys pass multiple keys (e.g. [\"ctrl\", \"c\"]). Valid key names: ctrl, shift, alt, cmd/meta, enter, esc, tab, space, backspace, delete, up, down, left, right, home, end, page_up, page_down, f1-f12, or any single character.",
  {
    keys: z.array(z.string()).describe("List of keys to press. Use lowercase names: ctrl, shift, alt, enter, tab, etc."),
  },
  async ({ keys }) => {
    const normalized = keys.map(normalizeKey);
    if (normalized.length === 1) {
      await client.sendCommand("press_key", { key: normalized[0] });
    } else {
      await client.sendCommand("hotkey", { keys: normalized });
    }
    return textOnly(`Pressed: ${normalized.join("+")}`);
  }
);

server.tool(
  "key_down",
  "On a desktop, press keys down without releasing them. Use with key_up to hold modifiers. Valid key names: ctrl, shift, alt, cmd/meta, enter, esc, tab, space, backspace, delete, up, down, left, right, home, end, page_up, page_down, f1-f12, or any single character.",
  {
    keys: z.array(z.string()).describe("List of keys to press down. Use lowercase names: ctrl, shift, alt, etc."),
  },
  async ({ keys }) => {
    const normalized = keys.map(normalizeKey);
    for (const k of normalized) {
      await client.sendCommand("key_down", { key: k });
    }
    return textOnly(`Key down: ${normalized.join("+")}`);
  }
);

server.tool(
  "key_up",
  "On a desktop, release keys that were previously pressed down with key_down. Valid key names: ctrl, shift, alt, cmd/meta, enter, esc, tab, space, backspace, delete, up, down, left, right, home, end, page_up, page_down, f1-f12, or any single character.",
  {
    keys: z.array(z.string()).describe("List of keys to release. Use lowercase names: ctrl, shift, alt, etc."),
  },
  async ({ keys }) => {
    const normalized = keys.map(normalizeKey);
    for (const k of normalized) {
      await client.sendCommand("key_up", { key: k });
    }
    return textOnly(`Key up: ${normalized.join("+")}`);
  }
);

server.tool(
  "type",
  "On a desktop, type text content into the currently focused input field.",
  {
    text: z.string().describe("The text content to type."),
  },
  async ({ text }) => {
    await client.sendCommand("type_text", { text });
    const preview = text.length > 50 ? text.slice(0, 50) + "..." : text;
    return textOnly(`Typed: "${preview}"`);
  }
);

server.tool(
  "hold_key",
  "On a desktop, hold keys down for a specified duration then release. Valid key names: ctrl, shift, alt, cmd/meta, enter, esc, tab, space, backspace, delete, up, down, left, right, home, end, page_up, page_down, f1-f12, or any single character.",
  {
    keys: z.array(z.string()).describe("List of keys to hold down. Use lowercase names: ctrl, shift, alt, etc."),
    duration: z.number().describe("Duration in seconds."),
  },
  async ({ keys, duration }) => {
    const normalized = keys.map(normalizeKey);
    for (const k of normalized) {
      await client.sendCommand("key_down", { key: k });
    }
    await new Promise((resolve) => setTimeout(resolve, duration * 1000));
    for (const k of [...normalized].reverse()) {
      await client.sendCommand("key_up", { key: k });
    }
    return textOnly(`Held ${normalized.join("+")} for ${duration}s`);
  }
);

// ================================================================
// Mouse Actions
// ================================================================

server.tool(
  "mouse_move",
  "On a desktop, move the mouse cursor to specified coordinates.",
  {
    coordinate: zCoordinate,
  },
  async ({ coordinate }) => {
    const { x, y } = await toAbsolute(coordinate);
    await client.sendCommand("move_cursor", { x, y });
    return textOnly(`Moved cursor to [${coordinate[0]}, ${coordinate[1]}]`);
  }
);

server.tool(
  "click",
  "On a desktop, perform mouse click at specified coordinates.",
  {
    coordinate: zCoordinate.optional().describe("(x, y) coordinates normalized to [0, 1000]."),
    button: z.enum(["left", "right", "middle"]).default("left").describe("Mouse button to click."),
    clicks: z.union([z.literal(1), z.literal(2), z.literal(3)]).default(1).describe(
      "Number of clicks: 1=single, 2=double, 3=triple."
    ),
  },
  async ({ coordinate, button, clicks }) => {
    const btn = button;
    const n = clicks;
    let abs = null;
    if (coordinate) {
      abs = await toAbsolute(coordinate);
    }

    if (n === 2 && btn === "left") {
      await client.sendCommand("double_click", abs ? { x: abs.x, y: abs.y } : {});
    } else if (btn === "middle") {
      if (abs) await client.sendCommand("move_cursor", { x: abs.x, y: abs.y });
      for (let i = 0; i < n; i++) {
        await client.sendCommand("mouse_down", { button: "middle" });
        await client.sendCommand("mouse_up", { button: "middle" });
      }
    } else {
      const cmd = btn === "right" ? "right_click" : "left_click";
      const params = abs ? { x: abs.x, y: abs.y } : {};
      for (let i = 0; i < n; i++) {
        await client.sendCommand(cmd, params);
      }
    }

    const coordLabel = coordinate ? ` at [${coordinate[0]}, ${coordinate[1]}]` : "";
    return textOnly(`Clicked (${btn}, ${n}x)${coordLabel}`);
  }
);

server.tool(
  "drag",
  "On a desktop, drag the mouse from start to end coordinates. Uses mouse_down + move_cursor + mouse_up for reliable cross-platform dragging.",
  {
    coordinate: zCoordinate.describe("Ending (x, y) coordinates, normalized to [0, 1000]."),
    start_coordinate: z.array(z.number()).length(2).optional().describe(
      "Starting (x, y) coordinates, normalized to [0, 1000]."
    ),
    button: z.enum(["left", "right", "middle"]).default("left").describe(
      "Mouse button to hold while dragging."
    ),
  },
  async ({ coordinate, start_coordinate, button }) => {
    const start = start_coordinate
      ? await toAbsolute(start_coordinate)
      : null;
    const end = await toAbsolute(coordinate);
    if (start) {
      await client.sendCommand("mouse_down", { x: start.x, y: start.y, button });
    } else {
      await client.sendCommand("mouse_down", { button });
    }
    await client.sendCommand("move_cursor", { x: end.x, y: end.y });
    await client.sendCommand("mouse_up", { x: end.x, y: end.y, button });
    const startLabel = start_coordinate
      ? `[${start_coordinate[0]}, ${start_coordinate[1]}]`
      : "current";
    return textOnly(
      `Dragged (${button}) from ${startLabel} to [${coordinate[0]}, ${coordinate[1]}]`
    );
  }
);

server.tool(
  "mouse_down",
  "On a desktop, press the mouse button without releasing.",
  {
    button: z.enum(["left", "right", "middle"]).default("left").describe("Mouse button to press."),
  },
  async ({ button }) => {
    await client.sendCommand("mouse_down", { button });
    return textOnly(`Mouse down: ${button}`);
  }
);

server.tool(
  "mouse_up",
  "On a desktop, release the mouse button.",
  {
    button: z.enum(["left", "right", "middle"]).default("left").describe("Mouse button to release."),
  },
  async ({ button }) => {
    await client.sendCommand("mouse_up", { button });
    return textOnly(`Mouse up: ${button}`);
  }
);

server.tool(
  "scroll",
  "On a desktop, scroll in a specified direction by a specified amount.",
  {
    direction: z.enum(["up", "down", "left", "right"]).describe("The direction to scroll."),
    amount: z.number().describe("Number of scroll units."),
    coordinate: zCoordinate.optional().describe(
      "(x, y) coordinates. If provided, cursor moves here before scrolling."
    ),
  },
  async ({ direction, amount, coordinate }) => {
    if (coordinate) {
      const { x, y } = await toAbsolute(coordinate);
      await client.sendCommand("move_cursor", { x, y });
    }
    await client.sendCommand("scroll_direction", { direction, clicks: amount });
    const pos = coordinate ? ` at [${coordinate[0]}, ${coordinate[1]}]` : "";
    return textOnly(`Scrolled ${direction} ${amount}${pos}`);
  }
);

// ================================================================
// Utility Actions
// ================================================================

server.tool(
  "wait",
  "On a desktop, pause execution for a specified duration.",
  {
    duration: z.number().describe("Time in seconds to wait."),
  },
  async ({ duration }) => {
    await new Promise((resolve) => setTimeout(resolve, duration * 1000));
    return textOnly(`Waited ${duration}s`);
  }
);

server.tool(
  "screenshot",
  "On a desktop, take a screenshot. Optionally save the image to a path on the VM.",
  {
    save_path: z.string().optional().describe(
      "Absolute file path on the VM to save the screenshot (e.g. C:\\\\tmp\\\\shot.png). " +
      "If omitted, the screenshot is returned as base64 only without saving to disk."
    ),
  },
  async ({ save_path }) => {
    if (save_path !== undefined) {
      const lastSep = Math.max(save_path.lastIndexOf("/"), save_path.lastIndexOf("\\"));
      if (lastSep <= 0) {
        return {
          content: [{ type: "text", text: `Error: invalid save_path "${save_path}" — must be an absolute path with a parent directory.` }],
          isError: true,
        };
      }
      const parentDir = save_path.slice(0, lastSep);
      const dirCheck = await client.sendCommand("directory_exists", { path: parentDir }).catch(() => null);
      if (!dirCheck || !dirCheck.exists) {
        return {
          content: [{ type: "text", text: `Error: parent directory "${parentDir}" does not exist on the VM.` }],
          isError: true,
        };
      }
    }

    const { base64, mimeType } = await client.screenshot();

    if (save_path !== undefined) {
      await client.sendCommand("write_bytes", { path: save_path, content_b64: base64 });
      return {
        content: [
          { type: "text", text: `Screenshot captured and saved to ${save_path}` },
          { type: "image", data: base64, mimeType },
        ],
      };
    }

    return {
      content: [
        { type: "text", text: "Screenshot captured" },
        { type: "image", data: base64, mimeType },
      ],
    };
  }
);

server.tool(
  "cursor_position",
  "On a desktop, get the current cursor position.",
  {},
  async () => {
    const result = await client.sendCommand("get_cursor_position");
    const pos = result.position;
    const norm = await toNormalized(pos.x, pos.y);
    return textOnly(`Cursor at [${norm[0]}, ${norm[1]}]`);
  }
);

// ------------------------------------------------------------------
// Start stdio transport
// ------------------------------------------------------------------
const transport = new StdioServerTransport();
await server.connect(transport);
