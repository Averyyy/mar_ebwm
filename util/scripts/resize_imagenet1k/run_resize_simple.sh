#!/bin/bash

# Simple nohup command to run ImageNet resize with 8 workers
# More conservative for SLURM login nodes

echo "Starting ImageNet resize with 8 workers using nohup..."
echo "Log file: resize_imagenet_$(date +%Y%m%d_%H%M%S).log"
echo "Start time: $(date)"

# Create logs directory
mkdir -p logs

nohup python3 resize_imagenet.py \
    --src_dir /work/nvme/belh/aqian1/imagenet-1k/ \
    --dst_dir /work/hdd/bdta/aqian1/data/imagenet-1k-64/ \
    --num_workers 8 \
    --batch_size 200 \
    > logs/resize_imagenet_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > logs/resize_imagenet.pid

echo "Job started with PID: $!"
echo "Monitor with: tail -f logs/resize_imagenet_*.log"
echo "Kill with: kill $!"