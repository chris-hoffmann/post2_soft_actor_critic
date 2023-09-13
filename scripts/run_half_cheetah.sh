#!/bin/bash

seeds="1 2 3"
for s in $seeds
do
    python train_sac.py \
        --seed $s \
        --env HalfCheetah-v4 \
        --time_steps 200_000 \
        --learning_starts 2_000 \
        --max_episode_steps 150 
done