#!/bin/bash

# Parameters
kernels=(5)
n_channels=(64)
n_CNNlayers=(5)
thetas=(0.0)
window_duration_percentile=(10 20 30 40 50 60 70 80 90 100)

device=3
padding="repeat"
python_script="train.py"
experiment_name="no_split"
num_epochs=100
match=0
batch_size=512

# Iterate over combinations
for theta in "${thetas[@]}"; do
    for channels in "${n_channels[@]}"; do
        for layers in "${n_CNNlayers[@]}"; do
            for kernel in "${kernels[@]}"; do
                for window_percentile in "${window_duration_percentile[@]}"; do

                    # Build the argument list
                    args="--experiment_name $experiment_name \
                          --padding $padding \
                          --num_epochs $num_epochs \
                          --window_duration_percentile $window_percentile \
                          --theta $theta \
                          --kernel_size $kernel \
                          --n_channels $channels \
                          --n_CNNlayers $layers \
                          --device $device \
                          --match $match \
                          --batch_size $batch_size"

                    echo "Running $python_script with:"
                    echo "$args"
                    python "$python_script" $args
                    echo "---------------------------------------------"

                done
            done
        done
    done
done
