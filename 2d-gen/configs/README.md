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

`standalone DreamBooth` config for `bash scripts/run_dreambooth_sd15.sh <config.yaml> [accelerate args...]`:

- `model.pretrained_model_name_or_path`: local SD1.5 diffusers model directory
- `model.pretrained_vae_model_name_or_path`: optional local VAE directory
- `model.revision`
- `model.variant`
- `model.local_files_only`
- `data.instance_data_dir`
- `data.instance_prompt`
- `data.class_data_dir`: required when `data.with_prior_preservation` is `true`
- `data.class_prompt`: required when `data.with_prior_preservation` is `true`
- `data.with_prior_preservation`
- `data.num_class_images`
- `data.resolution`
- `data.center_crop`
- `data.image_interpolation_mode`
- `train.output_dir`
- `train.seed`
- `train.train_batch_size`
- `train.sample_batch_size`
- `train.num_train_epochs`
- `train.max_train_steps`
- `train.checkpointing_steps`
- `train.checkpoints_total_limit`
- `train.resume_from_checkpoint`
- `train.gradient_accumulation_steps`
- `train.gradient_checkpointing`
- `train.learning_rate`
- `train.scale_lr`
- `train.lr_scheduler`
- `train.lr_warmup_steps`
- `train.lr_num_cycles`
- `train.lr_power`
- `train.dataloader_num_workers`
- `train.max_grad_norm`
- `train.allow_tf32`
- `train.mixed_precision`
- `train.prior_generation_precision`
- `train.enable_xformers_memory_efficient_attention`
- `train.noise_offset`
- `train.prior_loss_weight`
- `train.snr_gamma`
- `train.optimizer.use_8bit_adam`: currently unsupported and must remain `false`
- `train.optimizer.adam_beta1`
- `train.optimizer.adam_beta2`
- `train.optimizer.adam_weight_decay`
- `train.optimizer.adam_epsilon`
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
