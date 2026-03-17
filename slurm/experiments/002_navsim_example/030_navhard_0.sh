#!/usr/bin/bash
# Run this script in navsimv2.2 conda environment

source $LEAD_PROJECT_ROOT/slurm/init.sh

export CHECKPOINT_DIR=$LEAD_PROJECT_ROOT/outputs/training/002_navsim_example/010_postrain32_0/260316_120537/
export CHECKPOINT_FILE=model_0060.pth

evaluate_navhard --partition=2080-galvani
