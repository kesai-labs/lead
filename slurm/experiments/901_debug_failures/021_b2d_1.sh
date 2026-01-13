#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/733_scaled_regnety/011_postrain32_1/251025_182331

evaluate_bench2drive220
