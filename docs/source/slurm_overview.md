# Slurm Wrapper Overview

```{note}
This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.
```

LEAD includes an optional, minimal SLURM wrapper for running experiments on HPC clusters. The design is opinionated but has been used in our research workflow and is kept separate from the core codebase.

## Why Use This Wrapper?

The wrapper addresses several practical needs when running ML experiments on HPC clusters:

- **Simplified execution**: Submit training runs, evaluations, and restarts across multiple seeds without manual job submission
- **Consistent organization**: Generates unified names for SLURM jobs, WandB experiments, and output directories
- **Parallel execution**: Run multiple training and evaluation jobs simultaneously
- **Multi-cluster compatibility**: Supports different clusters with varying partition names

The core principle: **one experiment = one bash script**. Each script is version-controlled for reproducibility.

## How It Works: A Complete Example

Here's a typical workflow, starting with pre-training a model from scratch.

### Step 1: Creating Your First Training Script

A minimal pre-training script from [slurm/experiments/001_example/000_pretrain1_0.sh](https://github.com/autonomousvision/lead/blob/main/slurm/experiments/001_example/000_pretrain1_0.sh):

```{code-block} bash
:linenos:

#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"

train --cpus-per-task=32 --partition=a100-galvani --time=3-00:00:00 --gres=gpu:4
```

**Naming convention**: Scripts follow the pattern `slurm/experiments/<exp_id>_<exp_name>/<step_id>_<step_name>_<seed>.sh`. This creates a hierarchy: experiments contain multiple steps (pre-training, fine-tuning, evaluation), each potentially run with different random seeds.

**Line-by-line breakdown**:

1. **Line 1**: Standard bash script header
2. **Line 3**: Sources [slurm/init.sh](https://github.com/autonomousvision/lead/blob/main/slurm/init.sh), which sets up environment variables and defines helper functions like `train`
3. **Line 5**: Configures model architecture via environment variables (matching options in [lead/training/config_training.py](https://github.com/autonomousvision/lead/blob/main/lead/training/config_training.py))
4. **Line 7**: Launches training with SLURM parameters (CPUs, GPUs, time limit, partition name)

### Step 2: Running Your Experiment

Execute the script:

```bash
bash slurm/experiments/001_example/000_pretrain1_0.sh
```

The wrapper creates an organized output directory:

```bash
outputs/training/001_example/000_pretrain1_0/<year><month><day>_<hour><minute><second>
```

Logs, checkpoints, and metrics are saved to this timestamped directory for comparison and reproducibility.

## What's Next?

The same principles apply throughout the research pipeline:

- **[Data Collection](slurm_data_collection.md)**: Orchestrate CARLA simulation jobs to gather training data across the cluster
- **[Training](slurm_training.md)**: Run pre-training, fine-tuning, and multi-seed experiments with organized outputs
- **[Evaluation](slurm_evaluation.md)**: Test trained models across different scenarios and datasets

Each workflow uses the same script-based approach: one script per experiment step, version-controlled and automatically organized.
