#!/usr/bin/bash

source $LEAD_PROJECT_ROOT/slurm/init.sh

export CHECKPOINT_DIR=outputs/training/002_navsim_example/010_postrain32_0/251112_003238/model_0060.pth

evaluate_navhard --partition=2080-galvani
