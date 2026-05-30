#!/bin/bash
set -u
export ANTHROPIC_BASE_URL=https://openrouter.ai/api
export ANTHROPIC_AUTH_TOKEN='***REDACTED-FOR-DOC-ILLUSTRATION***'
export ANTHROPIC_API_KEY=''
cd /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746
prompt=$(cat /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/prompt.txt)
echo "$prompt" | /usr/local/bin/claude -p - --output-format stream-json --verbose --mcp-config /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/mcp_config.json --model anthropic/claude-sonnet-4-6 --max-turns 20 --dangerously-skip-permissions 2>/home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/stderr.log >/home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/transcript.jsonl
echo $? > /home/user/.ale/claude-code/cc_sonnet__anthropic-claude-sonnet-4-6__demo__demo_desktop_note_linux__v0__20260518_221746/done.marker
