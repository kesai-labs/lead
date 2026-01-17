#!/usr/bin/bash

source slurm/init.sh

export TRAINING_CONFIG="$TRAINING_CONFIG horizontal_fov_reduction=250"

train --cpus-per-task=32 --partition=a100-galvani --time=3-00:00:00 --gres=gpu:4
