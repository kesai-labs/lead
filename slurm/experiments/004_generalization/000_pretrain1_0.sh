#!/usr/bin/bash

source slurm/inits.sh

train --cpus-per-task=64 --partition=L40Sday --time=4-00:00:00 --gres=gpu:4
