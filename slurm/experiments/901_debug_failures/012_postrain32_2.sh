#!/usr/bin/bash

source slurm/init.sh

export TRAINING_CONFIG="$TRAINING_CONFIG image_architecture=regnety_032 lidar_architecture=regnety_032"
export TRAINING_CONFIG="$TRAINING_CONFIG use_planning_decoder=true"
posttrain outputs/training/733_scaled_regnety/000_pretrain1_0/251018_092144

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
