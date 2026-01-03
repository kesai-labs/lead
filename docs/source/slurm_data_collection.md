# Data Collection with SLURM

```{note}
This is a completely optional feature. The SLURM integration is designed for users with access to HPC clusters who want to scale their experiments efficiently. All functionality can also be run locally without SLURM.
```

Before training models, data collection is required. The data collection system orchestrates CARLA simulation jobs across the cluster, handling job submission, failure monitoring, and automatic resubmission of crashed routes.

## Preparing Your Data Collection Run

Configure the following settings in [slurm/data_collection/collect_data.py](https://github.com/autonomousvision/lead/blob/main/slurm/data_collection/collect_data.py):

- **`repetitions`**: Number of times to run each route with different random seeds (for data diversity)
- **`partitions`**: Cluster partition names (e.g., `gpu-2080ti`, `a100-galvani`)
- **`dataset_name`**: Descriptive name for the dataset (e.g., `carla_leaderboard2_train`)

Two additional config files control the collection process:

- [slurm/configs/max_num_parallel_jobs_collect_data.txt](https://github.com/autonomousvision/lead/blob/main/slurm/configs/max_num_parallel_jobs_collect_data.txt): Maximum number of parallel jobs (adjust based on cluster capacity)
- [slurm/configs/max_sleep.txt](https://github.com/autonomousvision/lead/blob/main/slurm/configs/max_sleep.txt): Delay between starting the CARLA server and Python client (prevents connection timing issues)

## Launching the Collection

Log into the cluster login node and start collection:

```bash
python3 slurm/data_collection/collect_data.py
```

The script creates a structured output directory:

```
data/carla_leaderboard2
├── data     # Sensor data storage
├── results  # Results JSON files
├── scripts  # Generated SLURM bash scripts
├── stderr   # SLURM stderr logs
└── stdout   # SLURM stdout logs
```

**Note**: Data collection can take up to 2 days on 90 GPUs for 9000 routes. Run the script inside `screen` or `tmux` to prevent interruption from SSH disconnections.

## Monitoring Your Collection

Check collection progress and identify failures:

```bash
python3 slurm/data_collection/print_collect_data_progress.py
```

Update the `root` variable in the script to point to your data directory.

```{note}
Failure rates below 10% are typical and primarily caused by simulation crashes or hardware issues.
Some scenario types may exhibit higher failure rates (around 50%), which indicates limitations in the expert's policy for those specific situations. This is expected behavior—as long as most scenarios maintain failure rates below 10%, the dataset quality remains sufficient for training.
```

## Cleaning Up Failed Routes

Remove corrupted or incomplete data after collection completes:

```bash
python3 slurm/data_collection/delete_failed_routes.py
```

This filters the dataset to only successfully collected routes.

```{warning}
This cleanup step is optional. The training pipeline filters out failed routes automatically.
Examining failed routes can reveal expert policy biases and data collection issues.
```

## What's Next?

With collected data, proceed to:
- **[Training](slurm_training.md)**: Train models using the collected data with automatic checkpointing and multi-seed support
- **[Evaluation](slurm_evaluation.md)**: Test trained models across scenarios and benchmark performance

The SLURM wrapper maintains consistent organization throughout the pipeline.
