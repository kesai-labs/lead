#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/746_hold_out_town13/010_postrain32_0/251104_090558

evaluate_town13
