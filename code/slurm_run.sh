#!/bin/bash
set -x

PARTITION=$1
NUM_NODES=1
NUM_GPUS_PER_NODE=1
CFG_PATH=$2
SCAN_ID=$3
PORT=$4

srun -p ${PARTITION} \
    -N ${NUM_NODES} \
    --gres=gpu:${NUM_GPUS_PER_NODE} \
    --cpus-per-task=4 \
    -t 5-00:00:00 \
    python training/exp_runner.py --conf $CFG_PATH --scan_id $SCAN_ID --port $PORT