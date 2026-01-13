#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/746_hold_out_town13/011_postrain32_1/251104_092049

evaluate_longest6
