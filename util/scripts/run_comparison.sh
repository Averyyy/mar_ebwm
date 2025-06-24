#!/bin/bash

# Activate environment
source activate mar_gh200

# Change to project directory
cd /work/hdd/bdta/aqian1/mar_ebwm

# Run comparison script
python compare_checkpoints.py --checkpoint1 "/work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-a-0.01-m-1/checkpoint-last.pth" --checkpoint2 "/work/hdd/bdta/aqian1/mar_ebwm/output/mar-base-energy-lr_1e-4-alpha_0.009-mult_0.03/checkpoint-last.pth" --output_dir "./checkpoint_comparison" --num_images 8 --class_id 1 --num_iter 64 --cfg 1.0 --temperature 1.0 --seed 42 --use_ema

echo "Comparison Finished" 