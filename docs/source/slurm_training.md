# Training with SLURM

```{note}
This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.
```

With your dataset collected, the next step is training models. The SLURM wrapper handles pre-training, fine-tuning, and multi-seed experiments with organized output directories and checkpoint resumption.

## A Complete Training Pipeline

A full example pipeline is available at [slurm/experiments/001_example](https://github.com/autonomousvision/lead/tree/main/slurm/experiments/001_example), demonstrating the workflow from initial training to evaluation-ready models.

## Step 1: Pre-training From Scratch

Train your perception backbone on the collected data:

```bash
bash slurm/experiments/001_example/000_pretrain1_0.sh
```

This starts a pre-training job with random seed `0` (indicated by the last digit in the script name). Output is automatically organized:

```
outputs/training/001_example/000_pretrain1_0/<year><month><day>_<hour><minute><second>
```

Checkpoints, logs, and metrics are saved to this directory for tracking training progress.

## Step 2: Post-training (Fine-tuning)

After pre-training completes, fine-tune by adding components like a planning decoder. Start post-training with seed `2`:

```bash
bash slurm/experiments/001_example/012_postrain32_2.sh
```

The script structure:

```{code-block} bash
:linenos:

#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"   # Same as in pre-training
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"                                       # Add a planner on top
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144                                                # Load the pre-trained weights

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
```

**Key points**:
- **Lines 5-6**: Model architecture configuration, consistent with pre-training plus new components
- **Line 7**: The `posttrain` function (defined in [slurm/init.sh](https://github.com/autonomousvision/lead/blob/main/slurm/init.sh)) helps the training script with loading the pre-trained weights
- **Line 9**: SLURM parameters for the fine-tuning job

## Handling Crashes: Automatic Resume

Training jobs crash due to hardware failures, time limits, or OOM errors. Recovery requires adding one line:

```{code-block} bash
:linenos:
:emphasize-lines: 8

#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"
posttrain outputs/training/001_example/000_pretrain1_0/251018_092144
resume outputs/training/001_example/012_postrain32_2/251018_092144                                                  # Resume from this checkpoint

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
```

The `resume` function loads the latest checkpoint from the specified directory. Restart training by running the same script without modifications.

## What's Next?

With trained models, proceed to:
- **[Evaluation](slurm_evaluation.md)**: Test models across different scenarios and benchmark performance
