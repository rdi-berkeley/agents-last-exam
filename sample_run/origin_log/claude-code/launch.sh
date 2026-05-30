#!/bin/bash
rm -f /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/done.marker /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/claude.pid
setsid bash /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/run_claude.sh </dev/null >/dev/null 2>&1 &
CHILD=$!
echo "$CHILD" > /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/claude.pid
disown $CHILD 2>/dev/null || true
