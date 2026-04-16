# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

Read `ARCHITECTURE.md` for the current repository layout and module map.

## Project Environment Rules

- Before starting implementation work, first align with the user on which project environment or `venv` should be used. Do not assume the environment if the user has not specified it.
- When working in `2d-gen`, use the virtual environment at `/home/jupyter-wenkaihua/data3/kaihua.wen/code/graduation-project_link/kaihua.wen/venv/diffusers` by default.
- When working in `2d-ssl-seg`, activate via `source scripts/run_with_venv.sh`.

## Commands

## Test Execution Rules

- For work in `2d-gen`, always run test commands with `PYTHONPATH=2d-gen/src` so imports resolve against the repository source tree.
- Prefer running tests from the repository root with the environment set inline, for example: `PYTHONPATH=2d-gen/src pytest ...`
- Do not assume a specific virtual environment. If a command depends on one, first align with the user per Project Environment Rules above.

## Coding Conventions

- Python: 4-space indentation, PEP 8, `snake_case` functions/variables, `PascalCase` classes
- YAML configs: `lower_snake_case` keys, group related settings together
- No formatter/linter configured — keep changes small and consistent with nearby code

## Commit & Pull Request Guidelines

- Use conventional commits with short imperative subjects (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, etc.).
- Always include `fixes #<number>` or `closes #<number>` in the commit message when there is a related issue or PR.
- Never include unrelated local changes in a commit. Stage and commit only the files that belong to the current task.
- Claude may commit directly for small, low-risk, self-contained changes (docs, comments, formatting, typo fixes, narrowly scoped housekeeping).
- Prefer a PR for behavior changes, multi-file refactors, new features, non-trivial bug fixes, changes to training/evaluation logic, changes that affect public interfaces or configs. If unclear, prefer opening a PR.
- PRs should include:
  - A concise summary of changes and rationale.
  - Links to related issues (if any).
  - Key command(s) run and resulting metrics or logs.
  - Config files and output paths used for reproducibility.
  - If Claude creates the PR, note "Created by Claude" in the description.

## Security & Configuration

- Do not commit datasets, checkpoints, or large artifacts; keep them under `outputs/`.

## Progressive Disclosure: diffusers

Scope: use these docs when analyzing or implementing training code for diffusion/generation models based on diffusers pipelines.
- diffusers repo: `/home/jupyter-wenkaihua/data3/kaihua.wen/code/graduation-project_link/kaihua.wen/code/diffusers`

Use as a narrowing path — start at the highest-level doc that answers the question, drill down only when needed, stop at the first layer with enough context:

1. `docs/diffusers/training_mapping.md` — cross-model mental model and family split
2. `docs/diffusers/training_architecture.md` — trainer and adapter contract
3. `docs/diffusers/stable_diffusion.md`, `docs/diffusers/stable_diffusion_3.md`, `docs/diffusers/sdxl.md`, `docs/diffusers/flux.md`, `docs/diffusers/qwenimage.md` — family-specific behavior
4. `docs/diffusers/reference/` — upstream scripts and implementation details

## Tool Usage Rules

- NEVER use `sed`/`cat` to read a file or a range of a file. Always use the Read tool (use offset + limit for ranged reads).
- You MUST read every file you modify in full before editing.

## Git Safety Rules

### Staging and committing
- **Only commit files you changed in the current session.**
- Always use `git add <specific-file-paths>` — never `git add -A` or `git add .`.
- Run `git status` before committing and verify you are only staging your own files.

### Forbidden operations
These commands can destroy work and must not be used:
- `git reset --hard` — destroys uncommitted changes
- `git checkout .` — destroys uncommitted changes
- `git clean -fd` — deletes untracked files
- `git stash` — stashes all changes including other sessions' work
- `git add -A` / `git add .` — stages unrelated uncommitted work
- `git commit --no-verify` — bypasses required checks

### Rebase conflicts
- Resolve conflicts only in files you modified.
- If a conflict is in a file you did not modify, abort and ask the user.
- Never force push.
