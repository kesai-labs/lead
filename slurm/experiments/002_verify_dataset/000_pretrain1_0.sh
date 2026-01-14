 #!/usr/bin/bash

source slurm/init.sh

export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG"

train --cpus-per-task=64 --partition=L40Sday --time=3-00:00:00 --gres=gpu:4
