#!/usr/bin/bash

source slurm/init.sh

export TRAINING_CONFIG="$TRAINING_CONFIG use_planning_decoder=true orizontal_fov_reduction=250"
posttrain outputs/training/005_tfv5_rig/000_pretrain1_0/260117_121230

train --cpus-per-task=64 --partition=L40Sday --time=3-00:00:00 --gres=gpu:4
