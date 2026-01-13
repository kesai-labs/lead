#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/733_scaled_regnety/012_postrain32_2/251025_182334

evaluate_longest6
