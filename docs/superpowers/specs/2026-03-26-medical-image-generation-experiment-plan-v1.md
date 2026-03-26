# Medical Image Generation Experiment Plan V1

## Goal

Answer the main thesis question:

Given the same medical dataset and the same LoRA fine-tuning protocol, can different base text-to-image generation models produce synthetic pretraining data that leads to different downstream segmentation performance?

## Fixed Main Line

- Main variable: base text-to-image model
- Fixed medical LoRA dataset: the current thesis dataset already chosen by the user
- Fixed SSL method: VICReg
- Fixed downstream model: ResNet18 encoder + UNet decoder
- Fixed downstream evaluation metrics: Dice, IoU, HD95

The study is designed so that generation training remains the primary task, while SSL and segmentation act as the validation chain.

## Stage 0: Lock The Experimental Protocol

Before running large experiments, fix the comparison rules so the results remain interpretable.

Items to lock:

- 3 candidate base text-to-image models
- one unified LoRA training protocol
- one unified generation protocol
- one matched synthetic dataset size for each model
- one fixed VICReg training protocol
- one fixed downstream segmentation protocol

Expected outputs:

- experiment protocol table
- model shortlist
- run order and dependency list

## Stage 1: Generation-Centered Main Experiments

Goal: compare the generation models first and build the synthetic datasets needed for the closed-loop study.

For each base model:

1. Fine-tune the model on the same medical dataset with LoRA.
2. Save the LoRA weights and key checkpoints.
3. Generate synthetic medical images with the same generation procedure.
4. Build a matched synthetic dataset for later VICReg pretraining.
5. Record generation-side logs and representative outputs.

Artifacts to keep:

- LoRA weights
- key training logs
- representative generated image samples
- generation config used for that model
- synthetic dataset path for that model

Success criteria:

- all candidate models can be fine-tuned successfully
- all candidate models can produce usable medical images
- all candidate models can produce matched synthetic datasets for downstream study

## Stage 2: Minimum Closed-Loop Validation

Goal: quickly test whether generation model differences propagate to final segmentation performance.

For each synthetic dataset:

1. Run VICReg pretraining.
2. Extract the pretrained encoder.
3. Fine-tune the fixed downstream segmentation model.
4. Evaluate final performance with Dice, IoU, and HD95.

This stage should stay minimal and answer only the main question:

Do different generation model versions produce different final segmentation performance through the same SSL and segmentation chain?

## Stage 3: Required Baselines

These baselines are required:

1. Random initialization with no SSL pretraining.
2. Real-only VICReg pretraining followed by downstream segmentation.
3. Synthetic-only VICReg pretraining for each compared generation model version followed by downstream segmentation.

Optional extension:

4. Real plus synthetic VICReg pretraining.

This mixed-data setting is useful, but it is not required for the first closed thesis loop.

## Stage 4: Result Interpretation

Goal: explain why certain generation models are more useful than others.

Recommended supporting analyses:

1. Generation quality and medical realism comparison across generation model versions.
2. Correlation analysis between generation-side indicators and downstream segmentation performance.

This stage is for interpretation, not for adding new primary variables.

## Minimum Main Experimental Matrix

The minimum recommended matrix for the first thesis milestone is:

- 3 base text-to-image models
- 3 synthetic-only experiments: synthetic data -> VICReg -> segmentation
- 1 real-only baseline: real data -> VICReg -> segmentation
- 1 random initialization baseline: no SSL -> segmentation

This produces 5 main result groups and is sufficient for a first closed-loop comparison.

## Recommended Execution Order

1. Finalize the 3 base models.
2. Complete LoRA fine-tuning for all 3 models.
3. Generate 3 matched synthetic datasets.
4. Run one full synthetic-only pipeline first to validate the workflow.
5. Run the remaining synthetic-only pipelines.
6. Complete the real-only and random-init baselines.
7. Perform result analysis and prepare thesis-ready tables and figures.

This order reduces risk by validating the pipeline before committing full compute time.

## Deliverables By Stage

### Stage 1

- LoRA weights
- generation configs
- representative generated samples
- synthetic dataset folders

### Stage 2

- VICReg checkpoints
- extracted encoders
- segmentation checkpoints
- evaluation logs

### Stage 3

- baseline comparison table
- main result table

### Stage 4

- correlation plots
- qualitative generation examples
- thesis-ready analysis summary

## Risk Control

The main risk is not an insufficient number of models. The main risk is failing to establish a closed research loop early.

Therefore the priority order should remain:

1. keep the number of compared models small
2. keep the protocol fixed
3. validate one full pipeline early
4. obtain the first segmentation comparison results as soon as possible

## Scope Notes

This experiment plan intentionally excludes the following from the main line:

- data cleaning strategy
- broad SSL method comparison
- broad segmentation architecture comparison
- large LoRA hyperparameter sweeps
- synthetic and real mixing ratio as a main variable

These may appear later only as supplementary experiments or future work.
