# Evaluation with SLURM


```{note}
This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.
```

After training models, the next step is systematic evaluation across benchmark scenarios. The SLURM wrapper handles batch evaluation with automatic route distribution and progress tracking.

## Overview

A complete example pipeline is available at [slurm/experiments/001_example](https://github.com/autonomousvision/lead/tree/main/slurm/experiments/001_example), demonstrating training and evaluation workflows.

For reliable results, train three models with different random seeds and evaluate each once on both Bench2Drive and Longest6 v2. For higher confidence, evaluate each seed three times.

## Running Evaluations

Here's an example evaluation script [slurm/experiments/001_example/020_b2d_0.sh](https://github.com/autonomousvision/lead/blob/main/slurm/experiments/001_example/020_b2d_0.sh):

```{code-block} bash
:linenos:

#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/001_example/010_postrain32_0/251025_182327  # Point to the trained model checkpoint
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"                                # Override training parameters at test time (rarely needed)
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"                          # Override closed-loop evaluation parameters

evaluate_bench2drive220
```

**Line-by-line breakdown**:

- **Line 5**: Specifies the checkpoint directory from training
- **Line 6**: Optional override for training configuration at test time (rarely used)
- **Line 7**: Optional override for closed-loop evaluation parameters (e.g., simulation settings)
- **Line 9**: Launches evaluation using the `evaluate_bench2drive220` function, which starts a `screen` session and distributes SLURM jobs for each route

Runtime parameters for evaluations are configured in [slurm/configs](https://github.com/autonomousvision/lead/tree/main/slurm/configs).

## Tracking Results

WandB logging is supported for evaluation metrics:

![](../assets/eval_wandb.png)

This provides real-time tracking of route completion, success rates, and performance metrics across the evaluation run.
