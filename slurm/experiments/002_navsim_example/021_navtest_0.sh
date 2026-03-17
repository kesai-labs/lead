#!/usr/bin/bash
# Run this script in navsimv1.1 conda environment

source $LEAD_PROJECT_ROOT/slurm/init.sh

export CHECKPOINT_DIR=$LEAD_PROJECT_ROOT/outputs/training/002_navsim_example/011_postrain32_1/260316_120540
export CHECKPOINT_FILE=model_0060.pth

evaluate_navtest --partition=2080-galvani
