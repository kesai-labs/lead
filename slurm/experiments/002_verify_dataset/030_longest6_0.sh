#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/002_verify_dataset/010_postrain32_0/260115_220458
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"

evaluate_longest6
