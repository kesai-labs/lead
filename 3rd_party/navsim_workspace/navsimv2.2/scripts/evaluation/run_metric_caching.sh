#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100-galvani
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=long.nguyen@student.uni-tuebingen.de
#SBATCH --mem=200gb

TRAIN_TEST_SPLIT=navtest
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_v2.2

export NAVSIM_DEVKIT_ROOT="${PROJECT_DIR}/3rd_party/navsim_workspace/navsimv2.2"
export HYDRA_FULL_ERROR=1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH
