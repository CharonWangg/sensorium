#! /bin/bash
PYTHON=/home/charon/anaconda3/envs/sensorium/bin/python
# model
CFG=/home/charon/project/sensorium/run/configs/.py
SEED=7
GPUS=[0]

$PYTHON api/train.py --cfg=$CFG --seed=$SEED --gpu_ids=$GPUS
