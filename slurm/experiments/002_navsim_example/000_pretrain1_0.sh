#!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_navsim_data=true use_carla_data=false LTF=true epochs=61"

train --cpus-per-task=32 --partition=L40Sday --time=1-00:00:00 --gres=gpu:4
