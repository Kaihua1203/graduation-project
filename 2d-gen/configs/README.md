# Config Guide

`train` config:

- `model.family`: `stable_diffusion | sdxl | flux | qwenimage`
- `model.pretrained_model_name_or_path`: local diffusers model directory
- `model.revision`: optional local revision
- `model.variant`: optional diffusers variant
- `model.local_files_only`: bool
- `model.pretrained_vae_model_name_or_path`: optional local VAE path
- `data.train_manifest`
- `data.resolution`
- `data.center_crop`
- `data.random_flip`
- `data.image_interpolation_mode`
- `train.output_dir`
- `train.train_batch_size`
- `train.num_train_epochs`
- `train.learning_rate`
- `train.max_train_steps`
- `train.gradient_accumulation_steps`
- `train.lr_scheduler`
- `train.lr_warmup_steps`
- `train.gradient_checkpointing`
- `train.mixed_precision`
- `train.seed`
- `train.dataloader_num_workers`
- `train.checkpointing_steps`
- `train.resume_from_checkpoint`
- `train.optimizer.*`
  - `train.optimizer.use_8bit_adam` is currently unsupported and must remain `false`
- `train.lora.rank`
- `train.lora.alpha`
- `train.lora.dropout`
- `train.lora.target_modules`
- `validation.validation_prompt`
- `validation.num_validation_images`
- `validation.validation_epochs`
- `logging.report_to`
- `logging.logging_dir`
- `logging.project_name`
- `logging.experiment_name`
- `distributed.find_unused_parameters`

`bash scripts/run_train.sh <config.yaml> [accelerate args...]` forwards extra arguments to `accelerate launch`. Single-GPU launches can omit launcher flags, while multi-GPU launches must pass `--multi_gpu`, `--num_processes`, and a matching `CUDA_VISIBLE_DEVICES` list, for example `CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh configs/train_sd_lora_example.yaml --multi_gpu --num_processes 2`. Distributed launcher settings should be passed via `accelerate launch`, not stored in the YAML config.

`infer` config:

- `model.family`
- `model.pretrained_path`
- `model.lora_path`
- `infer.prompts_path`: single `.txt` file or directory of `.txt` files
- `infer.output_dir`
- `infer.num_inference_steps`
- `infer.guidance_scale`
- `infer.height`
- `infer.width`
- `infer.seed`

Single-file inference keeps the existing line-by-line behavior. Directory-based inference processes `.txt` files in lexicographic order and treats each file as one prompt.

`eval` config:

- `eval.real_image_dir`
- `eval.generated_image_dir`
- `eval.generated_manifest`
- `eval.output_path`
- `eval.batch_size`
- `eval.num_workers`
- `eval.inception_weights_path`
- `eval.clip_model_path`
