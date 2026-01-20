#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/002_verify_dataset/012_postrain32_2/260116_173343
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"

evaluate_longest6
