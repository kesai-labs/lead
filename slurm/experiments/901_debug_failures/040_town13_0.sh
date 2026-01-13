#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/733_scaled_regnety/010_postrain32_0/251025_182327
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG"

evaluate_town13
