# Unconditional Latent Diffusion Evaluation Spec

## Topic

This spec defines the evaluation worktree for standalone unconditional latent diffusion in `2d-gen`.

This worktree owns:

- `2d-gen/src/eval/run_evaluate_uncond.py`
- any small shared evaluation helpers needed for unconditional mode
- eval YAMLs
- tests for eval contract and filename alignment

## Goal

Implement an unconditional evaluation path that:

- computes the intended image-only metrics
- rejects prompt-dependent metrics
- enforces directory and filename alignment before metric execution
- stays independent from training internals

## Fixed Context

- evaluation consumes only `real_image_dir` and `generated_image_dir`
- evaluation assumes unconditional generation and therefore has no prompt contract
- generated outputs must already align one-to-one with real-image filenames
- evaluation should reuse existing image-only metric implementations where possible
- evaluation must stay outside the prompt-conditioned workflow assumptions

## Supported Metrics

The unconditional evaluation entrypoint should compute:

- `FID`
- `Inception Score`
- `CLIP-I`
- `Med-FID`
- `BiomedCLIP-I`

## Explicitly Removed Metrics

The unconditional evaluation path must not compute:

- `CLIP-T`
- `BiomedCLIP-T`

These metrics depend on prompt-image pairs and are not meaningful for unconditional generation.

## Entry Point

This worktree owns:

```text
2d-gen/src/eval/run_evaluate_uncond.py
```

If small helper utilities are needed for unconditional-mode validation or metric selection, keep them narrowly scoped to evaluation.

## Eval Input Contract

The unconditional eval config should require:

- `real_image_dir`
- `generated_image_dir`
- `batch_size`
- `num_workers`
- `inception_weights_path`
- `clip_model_path`
- `biomedclip_model_path`
- cache directories for real-image feature caching when desired

It should additionally validate:

- both image directories exist
- image counts match
- sorted filenames match exactly

If filenames do not align, evaluation must fail early with a clear error.

## Filename Alignment Contract

Evaluation must treat filename alignment as part of the public interface, not a best-effort convenience.

Required behavior:

1. list and sort files in both image directories
2. verify equal file counts
3. verify exact filename equality at each sorted position
4. fail before metric execution if any mismatch is found

This protects the meaning of pairwise metrics such as `CLIP-I` and `BiomedCLIP-I`.

## YAML Deliverables

Add dataset-specific eval YAMLs for:

- `lits`
- `m4raw_t1`
- `m4raw_t2`
- `m4raw_flair`

Each eval YAML should include:

- `real_image_dir`
- `generated_image_dir`
- `output_path`
- `batch_size`
- `num_workers`
- model paths for Inception, CLIP, and BiomedCLIP
- optional cache directories

## Interfaces To Other Worktrees

- evaluation depends only on generated image directories produced by inference
- evaluation must not depend on training checkpoints, optimizer state, or trainer internals
- evaluation should assume the inference worktree already honored filename alignment, but it must still verify it

## Testing Requirements

At minimum, this worktree should add or cover:

- unconditional eval rejecting mismatched filenames
- unconditional eval rejecting mismatched image counts
- unconditional eval selecting only the intended metrics
- unconditional eval avoiding `CLIP-T` and `BiomedCLIP-T`

If runtime constraints are tight, tests should use tiny fixtures and mocked metric backends where possible.

## Success Criteria

This worktree is complete when the repo can:

1. evaluate unconditional outputs with `FID`, `IS`, `CLIP-I`, `Med-FID`, and `BiomedCLIP-I`
2. fail fast on directory or filename mismatch
3. avoid all prompt-dependent metric paths
4. run ready-to-use eval YAMLs for `lits`, `m4raw_t1`, `m4raw_t2`, and `m4raw_flair`
