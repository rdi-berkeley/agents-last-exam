# Demo: Desktop Note

## Overview

Minimal desktop-UI demo task for validating external agents that should
interact with the Windows desktop rather than browser or shell tools.

The agent must open Notepad, type two exact lines, and save the file to the
Windows Desktop.

## Expected Output

The agent must create this file:

`E:\agenthle\agenthle_desktop_plugin_demo.txt`

with exactly:

```text
AgentHLE Desktop Plugin Demo
This note was created through the desktop UI.
```

## Evaluation

Scoring is fully local and deterministic:

- `1.0` if the file exists and content matches exactly, ignoring newline style
- partial credit if only some expected lines appear
- `0.0` if the file is missing or unreadable

## Why This Task Exists

This task is intended to validate desktop-control integrations such as the
OpenClaw `cua` plugin. In contrast to `demo_web_search`, it should not be
solvable through browser/web tools alone.
