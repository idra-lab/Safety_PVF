#!/bin/bash

# State size
state_sizes=(10)

# List of alpha to test
alphas=(0.01 0.1 0.2 0.3)
SOURCE_DIR="plots"

# later copy only files modified in the last time_limit minutes
time_limit=90


# Loop over the arguments
for state_size in "${state_sizes[@]}"; do
    for alpha in "${alphas[@]}"; do
        python3 learning_nn/learn_V.py $state_size $alpha

        TARGET_DIR="size_${state_size}_alpha_${alpha}_plots"
        PATTERN="*state_size${state_size}*"

        # Check if target folder exists
        if [ -d "$TARGET_DIR" ]; then
            echo "Target folder exists. Deleting its contents..."
            rm -rf "$TARGET_DIR"/*
        else
            echo "Target folder does not exist. Creating it..."
            mkdir -p "$TARGET_DIR"
        fi

        # Move contents from source to target
        echo "Copying contents from source to target..."
        # cp "$SOURCE_DIR"/$PATTERN "$TARGET_DIR"
        find "$SOURCE_DIR" -name "$PATTERN" -mtime -$time_limit -type f -exec cp {} "$TARGET_DIR" \;
    done
done