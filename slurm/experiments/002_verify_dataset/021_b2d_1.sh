#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/002_verify_dataset/011_postrain32_1/260115_220501
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"

evaluate_bench2drive220
