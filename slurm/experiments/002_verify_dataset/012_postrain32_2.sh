#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_planning_decoder=true"
posttrain outputs/training/002_verify_dataset/000_pretrain1_0/260114_113602

train --cpus-per-task=64 --partition=L40Sday --time=2-00:00:00 --gres=gpu:4
