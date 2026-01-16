#!/usr/bin/bash

source scripts/utils.sh

export TRAINING_CONFIG="$TRAINING_CONFIG use_planning_decoder=true"
posttrain outputs/training/734_scaled_resnet34/000_pretrain1_0/251018_093017

train --cpus-per-task=64 --partition=L40Sday --time=3-00:00:00 --gres=gpu:4
