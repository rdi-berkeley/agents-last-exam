# Tools

This directory defines the actions ALE Claw exposes to the model and the
runtime paths those actions use to reach the VM or local harness services.

## Key ideas

- **Typed tool surface:** the model acts through named tools rather than raw
  shell access alone.
- **Tool categories:** file operations, shell, web, vision, memory, and GUI.
- **Sandbox access:** VM tools use the remote desktop session.
- **Computer handler:** GUI actions such as screenshots, clicks, typing, and
  scrolling are exposed through the `computer` tool.

## What lives here

- file tools such as `read`, `write`, and `edit`
- shell execution via `exec`
- web access via `web_search` and `web_fetch`
- image analysis via `analyze_image`
- GUI control via `computer`

The registry that assembles and filters the final tool list lives here too, so
this directory is the best place to start if you want to change what the model
can call.

## Read these files first

- `tools.py`: canonical tool registry and assembly
- `tools_fs.py`: file tools
- `tools_shell.py`: shell tool
- `tools_web.py`: web tools
- `analyze_image.py`: image-analysis tool
- `computer_handler.py`: GUI action handling
- `fs_backends.py`: filesystem backend routing
