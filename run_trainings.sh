#!/bin/bash

# List of state sizes to train
state_sizes=(3)

# List of target types
target_types=(0 1 2)


# Loop over the arguments
for state_size in "${state_sizes[@]}"; do
    for target_type in "${target_types[@]}"; do
        python3 learning_nn/learn_V_supervised.py $state_size $target_type
    done
done