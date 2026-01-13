#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/733_scaled_regnety/010_postrain32_0/251025_182327

evaluate_longest6
