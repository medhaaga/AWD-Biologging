#!/bin/bash

# List of cutoff values
kernels=(5)
n_channels=(64)
n_CNNlayers=(5)
thetas=(0.7)
window_duration_percentile=(10 20 30 40 50 60 70 80 90 100)
device=3
padding="repeat"
python_script='train.py'
experiment_name='no_split'
num_epochs=100
match=0

# Iterate over combinations of arg1 and arg2 values

for theta in "${thetas[@]}"; do   
    for channels in "${n_channels[@]}"; do
        for layers in "${n_CNNlayers[@]}"; do
            for kernel in "${kernels[@]}"; do
                for window_percentile in "${window_duration_percentile[@]}"; do
                    echo "Running $python_script with --experiment_name $experiment_name --padding $padding --num_epochs $num_epochs --window_duration_percentile $window_percentile --theta $theta --kernel_size $kernel --n_channels $channels --n_CNNlayers $layers --device $device --match $match" 
                    python "$python_script" --experiment_name "$experiment_name" --padding "$padding" --num_epochs "$num_epochs" --window_duration_percentile "$window_percentile" --theta "$theta" --kernel_size "$kernel" --n_channels "$channels" --n_CNNlayers "$layers" --device "$device" --match "$match"
                    echo "---------------------------------------------"
                done
            done
        done
    done
done
