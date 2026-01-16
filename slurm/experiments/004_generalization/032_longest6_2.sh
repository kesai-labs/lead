#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/746_hold_out_town13/012_postrain32_2/251104_091633

evaluate_longest6
