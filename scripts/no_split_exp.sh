#!/bin/bash

# Parameters
kernels=(5)
n_channels=(64)
n_CNNlayers=(5)
thetas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
seeds=(0 1 2 3 4 5)

device=0
padding="repeat"
python_script="train.py"
experiment_name="no_split"
num_epochs=100
match=0
batch_size=512
window_duration_percentile=50

# Iterate over combinations
for theta in "${thetas[@]}"; do
    for channels in "${n_channels[@]}"; do
        for layers in "${n_CNNlayers[@]}"; do
            for kernel in "${kernels[@]}"; do
                for seed in "${seeds[@]}"; do

                    # Build the argument list
                    args="--experiment_name $experiment_name \
                          --padding $padding \
                          --num_epochs $num_epochs \
                          --window_duration_percentile $window_duration_percentile \
                          --theta $theta \
                          --kernel_size $kernel \
                          --n_channels $channels \
                          --n_CNNlayers $layers \
                          --device $device \
                          --match $match \
                          --batch_size $batch_size \
                          --seed $seed"

                    echo "Running $python_script with:"
                    echo "$args"
                    python "$python_script" $args
                    echo "---------------------------------------------"

                done
            done
        done
    done
done
