## 2026-03-29

- Flux LoRA multi-GPU training with `train_flux_lora-4gpu-1000.yaml` failed immediately, but the root cause was not OOM. The first hard failure in the tmux log was `TypeError: CombinedTimestepGuidanceTextProjEmbeddings.forward() missing 1 required positional argument: 'pooled_projection'`.
- The bug only surfaced on multi-GPU because `accelerate.prepare()` wrapped `self.transformer` in DDP. `FluxAdapter.sample_time_state()` was checking `self.transformer.config.guidance_embeds`, which works on single GPU but can fail after wrapping because the DDP wrapper does not expose `config` like the underlying transformer.
- Fix applied in `2d-gen/src/train/adapters/flux.py`: cache the model's `guidance_embeds` requirement during setup and resolve guidance requirements from the wrapped module safely. This keeps Flux on the correct `time_text_embed(timestep, guidance, pooled_projections)` path in both single-GPU and multi-GPU launches.
- Regression coverage added in `2d-gen/src/tests/test_adapter_validators.py` to ensure guidance detection still works when the transformer is accessed through a wrapper object instead of directly.

### Lessons learned

- Do not read model capability flags from objects that may later be wrapped by DDP, FSDP, or other accelerator containers. Cache invariant config values early or unwrap explicitly before reading them.
- When a multi-GPU job fails right after launch, inspect the first rank-local Python exception before suspecting OOM. Distributed teardown often produces secondary NCCL or SIGTERM noise that can hide the real cause.
