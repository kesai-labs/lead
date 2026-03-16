#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_navsim_data=true use_carla_data=false LTF=true use_planning_decoder=true"
posttrain outputs/training/002_navsim_example/000_pretrain1_0/260315_145840

train --cpus-per-task=32 --partition=L40Sday --time=1-00:00:00 --gres=gpu:4
