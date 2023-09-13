#!/bin/bash

seeds="1 2 3"
for s in $seeds
do
    python train_sac.py \
        --seed $s \
        --env_name InvertedPendulum-v4 \
        --time_steps 50_000 \
        --learning_starts 2_000 
done