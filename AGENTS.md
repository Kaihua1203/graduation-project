# Repository Guidelines

## Project Structure & Module Organization
Read `ARCHITECTURE.md` for the current repository layout and module map.

## Project Environment Rules
- Before starting implementation work, first align with the user on which project environment or `venv` should be used. Do not assume the environment if the user has not specified it.
- When working in the `2d-gen` repository, use the virtual environment located at `/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/diffusers`.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 naming.
- Use `snake_case` for functions/variables, `PascalCase` for classes.
- Config files are YAML; keep keys lower_snake_case and group related settings together.
- No formatter/linter is configured; keep changes small and consistent with nearby code.

## Test Execution Rules
- For work in `2d-gen`, always run test commands with `PYTHONPATH=2d-gen/src` so imports resolve against the repository source tree.
- Prefer running tests from the repository root with the environment set inline, for example: `PYTHONPATH=2d-gen/src pytest ...`
- Do not assume a specific virtual environment. If a command depends on one, first align with the user according to `Project Environment Rules`.

## Commit & Pull Request Guidelines
- Use conventional commits with short, imperative subjects:
  - Prefer the format: `<type>(codex): <short imperative summary>`.
  - Include `Codex` attribution in the commit body when a body is used.
  - Examples: `feat(codex): add vicreg config for lits`, `fix(codex): handle empty lits directory`, `docs(codex): clarify agent workflow`.
- When Codex completes any meaningful `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, or similar change, it should create a dedicated commit for that unit of work after verification. Push when the work is ready for delivery, review, or handoff; do not leave completed Codex work uncommitted without a clear reason.
- Never include unrelated local changes in a Codex commit. Stage and commit only the files that belong to the current Codex task.
- For substantial or higher-risk Codex tasks, default to spawning three sub-agents to reduce main-session context pressure and separate responsibilities. For small or clearly bounded tasks, the main agent may simplify this workflow when doing so is more efficient:
  - `reviewer`: review the produced changes for bugs, regressions, missing edge cases, and documentation gaps.
  - `tester`: add or update relevant tests, run the validation commands, and report the exact commands and outcomes.
  - `git agent`: handle staging only the Codex-owned files for this session, create the commit, push the branch, and prepare the PR content.
- The main agent remains responsible for final technical decisions, integrating feedback from the three sub-agents, and ensuring their outputs are consistent before commit and push.
- Codex may commit directly for small, low-risk, self-contained changes that do not materially change behavior, such as docs, comments, formatting, typo fixes, or narrowly scoped housekeeping edits, if the repository workflow allows it.
- Codex should prefer a PR for behavior changes, multi-file refactors, new features, non-trivial bug fixes, changes to training or evaluation logic, changes that affect public interfaces or configs, or any work that benefits from explicit review and traceability.
- If it is unclear whether direct commit or PR is more appropriate, prefer opening a PR.
- PRs should include:
  - A concise summary of changes and rationale.
  - Links to related issues (if any).
  - Key command(s) run and resulting metrics or logs.
  - Config files and output paths used for reproducibility.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or large artifacts; keep them under `outputs/`.
- If you add new data paths or secrets, document them in `2d-ssl-seg/README.md` and keep them out of git.

## solo-learn Progressive Disclosure
Scope in this repo:
- use `solo-learn` only for SSL pretraining and encoder/backbone weight export for downstream segmentation.
- solo-learn repo: `/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/solo-learn`

Read in order:
1. `docs/solo-learn/solo-learn-core.md`
2. `docs/solo-learn/methods_and_backbones.md`

## diffusers Generation Model Training Docs Progressive Disclosure
Scope in this repo:
- use these docs when analyzing or implementing training code for diffusion / generation models based on diffusers pipelines.
- diffusers repo: `/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/diffusers`

Disclosure path:
1. `docs/diffusers/README.md` for the entry point and reading strategy.
2. `docs/diffusers/training_mapping.md` for the cross-model mental model and family split.
3. `docs/diffusers/training_architecture.md` for the trainer and adapter contract.
4. `docs/diffusers/stable_diffusion.md`, `docs/diffusers/stable_diffusion_3.md`, `docs/diffusers/sdxl.md`, `docs/diffusers/flux.md`, `docs/diffusers/qwenimage.md` for family-specific behavior.
5. `docs/diffusers/reference/` for upstream scripts and implementation details.

Use these docs as a narrowing path, not as a mandatory fixed reading order:
- start at the highest-level doc that can answer the question
- drill down only when you need the next layer of detail
- stop at the first layer that gives enough context to make a correct change

## **CRITICAL** Tool Usage Rules **CRITICAL**
- NEVER use sed/cat to read a file or a range of a file. Always use the read tool (use offset + limit for ranged reads).
- You MUST read every file you modify in full before editing.

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
git add 

# 3. Commit
git commit -m "fix(codex): description"

# 4. Push (pull --rebase if needed, but NEVER reset/checkout)
git pull --rebase && git push
```

### If Rebase Conflicts Occur
- Resolve conflicts in YOUR files only
- If conflict is in a file you didn't modify, abort and ask the user
- NEVER force push
