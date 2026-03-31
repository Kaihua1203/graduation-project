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
  - Examples: `feat: add vicreg config for lits`, `fix: handle empty lits directory`.
- When Codex makes functional changes (e.g., adding/removing evaluation pipeline pieces, training algorithms, or other behavior changes), or performs substantial multi-file/code-volume edits, launch a `reviewer` agent to review code. Then create one commit for that work and push it in the github.
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
- start from the cross-model training docs first, then drill down into model-specific pipeline docs only when needed.
- diffusers repo: `/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/diffusers`

Read in order:
1. `docs/diffusers/training_mapping.md`
2. `docs/diffusers/training_architecture.md`

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
git commit -m "fix: description"

# 4. Push (pull --rebase if needed, but NEVER reset/checkout)
git pull --rebase && git push
```

### If Rebase Conflicts Occur
- Resolve conflicts in YOUR files only
- If conflict is in a file you didn't modify, abort and ask the user
- NEVER force push
