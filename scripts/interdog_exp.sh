#!/bin/bash

kernels=(3 5)
n_channels=(16 32 64)
n_CNNlayers=(2 3 4 5)
thetas=(0.0)
device=2
padding="repeat"
python_script='train.py'
experiment_name='interdog'
window_duration_percentile=50
num_epochs=100
create_class_imbalance=0 

# Iterate over combinations of arg1 and arg2 values

for theta in "${thetas[@]}"; do   
    for channels in "${n_channels[@]}"; do
        for layers in "${n_CNNlayers[@]}"; do
            for kernel in "${kernels[@]}"; do
                echo "Running $python_script with --experiment_name $experiment_name --padding $padding --num_epochs $num_epochs --window_duration_percentile $window_duration_percentile --theta $theta --kernel_size $kernel --n_channels $channels --n_CNNlayers $layers --device $device" 
                python "$python_script" --experiment_name "$experiment_name" --padding "$padding" --num_epochs "$num_epochs" --window_duration_percentile "$window_duration_percentile" --theta "$theta" --kernel_size "$kernel" --n_channels "$channels" --n_CNNlayers "$layers" --device "$device" 
                echo "---------------------------------------------"
            done
        done
    done
done
