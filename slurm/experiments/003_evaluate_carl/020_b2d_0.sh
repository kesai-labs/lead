#!/usr/bin/bash

source slurm/init.sh

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LEAD_CLOSED_LOOP_CONFIG="produce_demo_video=true"
export CHECKPOINT_DIR=outputs/checkpoints/CaRL

evaluate_carl_bench2drive
