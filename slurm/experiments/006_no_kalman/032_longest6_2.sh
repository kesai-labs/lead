#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/733_scaled_regnety/012_postrain32_2/251025_182334
export LEAD_CLOSED_LOOP_CONFIG="$LEAD_CLOSED_LOOP_CONFIG produce_frame_frequency=1 produce_debug_video=false produce_debug_image=false produce_demo_video=true produce_demo_image=false produce_grid_image=false produce_grid_video=true"
export LEAD_TRAINING_CONFIG="$LEAD_TRAINING_CONFIG use_kalman_filter_for_gps=false"

evaluate_longest6
