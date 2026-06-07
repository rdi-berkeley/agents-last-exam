# Development Rules

## Repository layout

```
agents-last-exam/
├── ale_run/                  Framework code (python -m ale_run is the entry point)
│   ├── agents/                 Agent deployers: claude_code, ale_claw, …
│   ├── base_interface/         The contracts: Provider / Executor / Deployer / Trajectory
│   ├── environments/           Providers (gcloud, static) + image registry
│   ├── executors/              Where a deployer runs: sandbox / local / docker
│   ├── orchestration/          Run lifecycle, config loader, factories
│   └── tasks/                  Task discovery + driver
├── tasks/                    Task packages, grouped by domain (demo/ has the templates)
├── configs/                  Reusable agent + environment configs (referenced by path)
├── selected_tasks/           Curated task lists (cli, full, unlicensed)
├── secret/                   .env + GCP key + per-judge eval keys (real values gitignored)
├── docs/                     Setup/task/extension guides + the docs/ale-docs-site/ HTML site
├── example_exp.yaml          The minimal experiment; start here
└── pyproject.toml            uv workspace; Python ≥3.12, <3.14
```

## Conversational Style

- Keep answers short and concise
- No emojis in commits, issues, PR comments, or code
- No fluff or cheerful filler text
- Technical prose only, be kind but direct (e.g., "Thanks @user" not "Thanks so much @user!")
- When the user asks a question, answer it first before making edits or running implementation commands.
- In docs (README, `docs/`, user-facing markdown), avoid em dashes (—) when possible; prefer periods, commas, colons, or parentheses instead.

## Code Quality

- Read files in full before making wide-ranging changes, before editing files you have not already fully inspected, and when the user asks you to investigate or audit something. Do not rely only on search snippets for broad changes.
- Single-line helper functions with a single call site are forbidden; inline them instead.
- Always ask before removing functionality or code that appears to be intentional.
- Do not preserve backward compatibility unless the user explicitly asks for it.
- Check installed package source (`.venv/lib/python*/site-packages/<pkg>` or the editable `submodules/cua/...` paths in `pyproject.toml`) for external API shapes instead of guessing.
- NEVER remove or downgrade code to fix errors from outdated dependencies; upgrade the dependency instead.

## Comments

- Do not over-comment. Code that reads clearly does not need a narrator.
- Never put the discussion, the analysis, the debate, the dead-end attempts, or the "why I rewrote this" history into code comments. That belongs in commit messages or PR descriptions; the codebase stays clean.
- A comment is justified when it records intent that the code itself cannot express: an invariant, a non-obvious constraint from an upstream system, a "do not touch this because X", a citation to a spec/issue. If a comment is just restating what the next line does, delete it.
- Same rule for docstrings: describe contract (inputs, outputs, side effects, raises), not implementation narrative.

## Commands

- This is a Python project managed with `uv`. Use `uv run ...` for one-offs and `uv sync --extra dev` to install.
- Lint/typecheck: `uv run ruff check ale tasks tests` (and `ruff format --check` if checking formatting). Fix all errors before committing.
- Tests: `uv run pytest tests/<path>` for targeted runs; `uv run python tests/smoke_hello.py` for the in-process smoke. Real-VM smokes under `tests/integration/` cost money — only run when the user asks.
- If you create or modify a test file, you MUST run that test file and iterate until it passes.
- When writing tests, run them, identify issues in either the test or implementation, and iterate until fixed.
- NEVER commit unless the user asks.

## GitHub: issues and PR comments

When posting issue/PR comments:

- Write the full comment to a temp file and use `gh issue comment --body-file` or `gh pr comment --body-file`
- Never pass multi-line markdown directly via `--body` in shell commands
- Preview the exact comment text before posting
- Post exactly one final comment unless the user explicitly asks for multiple comments
- If a comment is malformed, delete it immediately, then post one corrected comment
- Keep comments concise, technical, and in the user's tone

When closing issues via commit:

- Include `fixes #<number>` or `closes #<number>` in the commit message
- This automatically closes the issue when the commit is merged

## PR Workflow

- Analyze PRs without pulling locally first
- If the user approves: create a feature branch, pull PR, rebase on main, apply adjustments, commit, merge into main, push, close PR, and leave a comment in the user's tone
- You never open PRs yourself. We work in feature branches until everything is according to the user's requirements, then merge into main, and push.

## **CRITICAL** Git Rules for Parallel Agents **CRITICAL**

Multiple agents may work on different files in the same worktree simultaneously. You MUST follow these rules:

### Committing

- **ONLY commit files YOU changed in THIS session**
- ALWAYS include `fixes #<number>` or `closes #<number>` in the commit message when there is a related issue or PR
- NEVER use `git add -A` or `git add .` - these sweep up changes from other agents
- ALWAYS use `git add <specific-file-paths>` listing only files you modified
- Before committing, run `git status` and verify you are only staging YOUR files
- Track which files you created/modified/deleted during the session

### Forbidden Git Operations

These commands can destroy other agents' work:

- `git reset --hard` - destroys uncommitted changes
- `git checkout .` - destroys uncommitted changes
- `git clean -fd` - deletes untracked files
- `git stash` - stashes ALL changes including other agents' work
- `git add -A` / `git add .` - stages other agents' uncommitted work
- `git commit --no-verify` - bypasses required checks and is never allowed

### Safe Workflow

```bash
# 1. Check status first
git status

# 2. Add ONLY your specific files
git add ale/agents/foo/deployer.py
git add tests/smoke_foo.py

# 3. Commit
git commit -m "fix(agents): description"

# 4. Push (pull --rebase if needed, but NEVER reset/checkout)
git pull --rebase && git push
```

### If Rebase Conflicts Occur

- Resolve conflicts in YOUR files only
- If conflict is in a file you didn't modify, abort and ask the user
- NEVER force push

### User override

If the user instructions conflict with rules set out here, ask for confirmation that they want to override the rules. Only then execute their instructions.
