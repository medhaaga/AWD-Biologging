#!/bin/bash

# List of cutoff values
kernels=(5)
n_channels=(32)
n_CNNlayers=(5)
thetas=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
device=3
padding="repeat"
python_script='train.py'
experiment_name='interyear'
duration_percentile=50
num_epochs=100
create_class_imbalance=0 

# Iterate over combinations of arg1 and arg2 values

for theta in "${thetas[@]}"; do   
    for channels in "${n_channels[@]}"; do
        for layers in "${n_CNNlayers[@]}"; do
            for kernel in "${kernels[@]}"; do
                echo "Running $python_script with --experiment_name $experiment_name --padding $padding --num_epochs $num_epochs --duration_percentile $duration_percentile --theta $theta --kernel_size $kernel --n_channels $channels --n_CNNlayers $layers --device $device --create_class_imbalance $create_class_imbalance" 
                python "$python_script" --experiment_name "$experiment_name" --padding "$padding" --num_epochs "$num_epochs" --duration_percentile "$duration_percentile" --theta "$theta" --kernel_size "$kernel" --n_channels "$channels" --n_CNNlayers "$layers" --device "$device" --create_class_imbalance "$create_class_imbalance"
                echo "---------------------------------------------"
            done
        done
    done
done