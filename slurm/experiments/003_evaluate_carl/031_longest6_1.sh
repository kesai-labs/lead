#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/checkpoints/CaRL

evaluate_carl_longest6
