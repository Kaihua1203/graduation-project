# Config Guide

`train` config:

- `model.family`: `stable_diffusion | sdxl | flux | qwenimage`
- `model.pretrained_path`: local diffusers model directory
- `model.revision`: optional local revision
- `model.local_files_only`: bool
- `train.output_dir`
- `train.num_epochs`
- `train.batch_size`
- `train.learning_rate`
- `train.max_train_steps`
- `train.gradient_accumulation_steps`
- `train.mixed_precision`
- `train.seed`
- `train.image_size`
- `train.lora_rank`
- `train.lora_alpha`
- `train.lora_dropout`
- `train.target_modules`
- `data.train_manifest`

`infer` config:

- `model.family`
- `model.pretrained_path`
- `model.lora_path`
- `infer.prompts_path`
- `infer.output_dir`
- `infer.num_inference_steps`
- `infer.guidance_scale`
- `infer.height`
- `infer.width`
- `infer.seed`

`eval` config:

- `eval.real_image_dir`
- `eval.generated_image_dir`
- `eval.generated_manifest`
- `eval.output_path`
- `eval.batch_size`
- `eval.num_workers`
- `eval.inception_weights_path`
- `eval.clip_model_path`
