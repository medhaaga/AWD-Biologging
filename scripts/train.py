# System & OS

import sys
import os
import time
import json
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")


import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Script imports

from src.utils.train import (train_run)

from src.utils.io import (format_time,
                          get_results_path,
                          get_metadata_path,
                          get_video_labels_path,
                          get_audio_labels_path,
                          get_matched_data_path,
                          get_matched_metadata_path)

from src.methods.prediction_model import create_dynamic_conv_model

from src.utils.data import (setup_data_objects,
                            setup_multilabel_dataloaders,
                            get_exp_filter_profiles, 
                            create_artificial_class_imbalance)

from src.data_prep.data_prep_utils import combined_annotations

from config.settings import (VECTRONICS_METADATA_PATH,
                             VECTRONICS_VIDEO_ANNOTATIONS_PATH,
                             VECTRONICS_AUDIO_ANNOTATIONS_PATH,
                             id_mapping,
                             COLLAPSE_BEHAVIORS_MAPPING,
                             BEHAVIORS)

from src.methods.conformal_prediction import *
from src.utils.conformal import *


##############################################
# Arguments
##############################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default='no_split', choices=['no_split', 'interdog', 'interyear', 'interAMPM'])
    parser.add_argument("--kernel_size", type=int, default=5, help="size fo kernel for CNN")
    parser.add_argument("--n_channels", type=int, default=64, help="number of output channels for the first CNN layer")
    parser.add_argument("--n_CNNlayers", type=int, default=3, help="number of convolution layers")
    parser.add_argument("--window_duration_percentile", type=int, default=50, help="audio duration cutoff percentile")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=.0001)
    parser.add_argument("--weight_decay", type=float, default=.0001)
    parser.add_argument("--normalization", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train_test_split", type=float, default=0.2)
    parser.add_argument("--train_val_split", type=float, default=0.2)
    parser.add_argument("--filter_type", type=str, default='high')
    parser.add_argument("--padding", type=str, default='repeat', choices=['zeros', 'repeat'])
    parser.add_argument("--cutoff_frequency", type=float, default=0)
    parser.add_argument("--cutoff_order", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--match", type=int, default=0, help="should the matching be done or use pre-matched observations?")
    parser.add_argument("--min_duration", type=float, default=1.0, help="minimum duration of a behavior in seconds so that it is not discarded")
    parser.add_argument("--create_class_imbalance", type=int, default=0, help="whether to create class imbalance artificially")
    parser.add_argument("--class_imbalance_percent", type=float, default=0.01, help="percetage of feeding behavior in the imbalanced dataset")
    parser.add_argument("--alpha", type=float, default=0.05, help="coverage for RAPS is 1-alpha")
    parser.add_argument("--verbose", type=int, default=0, help="whether to print training logs")

    return parser


if __name__ == '__main__':

    # parse arguments
    parser = parse_arguments()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # experiment directory 
    dir = get_results_path(args.experiment_name, args.n_CNNlayers, args.n_channels, args.kernel_size, args.theta, args.window_duration_percentile)
    os.makedirs(dir, exist_ok=True)

    # train-test split profile
    train_filter_profile, test_filter_profile = get_exp_filter_profiles(args.experiment_name) 

    ##############################################
    # loading data and creating train/test split
    ##############################################

    if os.path.exists(get_metadata_path()):
        metadata = pd.read_csv(get_metadata_path()) # load metadata
    elif os.path.exists(VECTRONICS_METADATA_PATH):
        metadata = pd.read_csv(VECTRONICS_METADATA_PATH) # load metadata
    else:
        raise FileNotFoundError("The metadata not found.")

    if os.path.exists(VECTRONICS_VIDEO_ANNOTATIONS_PATH) and os.path.exists(VECTRONICS_AUDIO_ANNOTATIONS_PATH):
        all_annotations = combined_annotations(video_path=VECTRONICS_VIDEO_ANNOTATIONS_PATH, 
                                            audio_path=VECTRONICS_AUDIO_ANNOTATIONS_PATH,
                                            id_mapping=id_mapping) # load annotations 
    elif os.path.exists(get_video_labels_path()) and os.path.exists(get_audio_labels_path()):
        all_annotations = combined_annotations(video_path=get_video_labels_path(), 
                                            audio_path=get_audio_labels_path(),
                                            id_mapping=id_mapping) # load annotations 
    else:
        raise FileNotFoundError("The annottaions not found.")


    start = time.time()
    X_train, y_train, z_train, X_val, y_val, z_val, X_test, y_test, z_test, _ = setup_data_objects(metadata, 
                                                                                                    all_annotations, 
                                                                                                    COLLAPSE_BEHAVIORS_MAPPING, 
                                                                                                    BEHAVIORS, 
                                                                                                    args, 
                                                                                                    reuse_behaviors=BEHAVIORS,
                                                                                                    acc_data_path=get_matched_data_path(),
                                                                                                    acc_metadata_path=get_matched_metadata_path()) 
    
    
    if args.create_class_imbalance:
        X_train, y_train = create_artificial_class_imbalance(X_train, y_train, args.class_imbalance_percent)
        dir = os.path.join(dir, 'class_imbalance/')
        os.makedirs(dir, exist_ok=True)

    
    print("Class distribution")
    print("==========================")
    print(pd.DataFrame(np.unique(y_train, return_counts=True)[1]))
    print("")


    n_timesteps, n_features, n_outputs = X_train.shape[2], X_train.shape[1], len(np.unique(np.concatenate((y_train, y_val, y_test))))
    train_dataloader, val_dataloader, test_dataloader = setup_multilabel_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args)
    time_diff = time.time() - start

    print("")
    print(f'Creating data objects takes {time_diff:.2f} seconds.')
    print("")
    print('Shape of dataframes')
    print("==========================")
    print(f"Train: -- X: {train_dataloader.dataset.tensors[0].shape}, Y: {train_dataloader.dataset.tensors[1].shape}, Z: {z_train.shape}")
    print(f"Val: -- X: {val_dataloader.dataset.tensors[0].shape}, Y: {val_dataloader.dataset.tensors[1].shape}, Z: {z_val.shape}")
    print(f"Test: -- X: {test_dataloader.dataset.tensors[0].shape}, Y: {test_dataloader.dataset.tensors[1].shape}, Z: {z_test.shape}")

    #########################################
    #### Model, loss, and optimizer
    #########################################

    # Define the sequential model
    model = create_dynamic_conv_model(n_features, n_timesteps, n_outputs, 
                                        num_conv_layers=args.n_CNNlayers, 
                                        base_channels=args.n_channels, 
                                        kernel_size=args.kernel_size).to(device)

    print("")
    print("==================================")
    print(f"Number of trainable model paramters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_obj = train_run(model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader, args, device)
    model = train_obj['model']
    training_stats = train_obj['training_stats']


    #############################################
    ###### Save objects
    ##############################################

    torch.save(model, os.path.join(dir, 'model.pt'))
    json_training_stats_file = os.path.join(dir, 'training_stats.json')
    with open(json_training_stats_file, 'w') as f:
        json.dump(training_stats, f)

    # save true and predicted validation classes along with val metadata
    np.save(os.path.join(dir, 'val_true_classes.npy'),  train_obj['val_true_classes'])
    np.save(os.path.join(dir, 'val_predictions.npy'),  train_obj['val_predictions'])
    np.save(os.path.join(dir, 'val_scores.npy'),  train_obj['val_scores'])
    z_test.to_csv(os.path.join(dir, 'val_metadata.csv'))


    # save true and predicted validation classes along with val metadata
    np.save(os.path.join(dir, 'test_true_classes.npy'),  train_obj['test_true_classes'])
    np.save(os.path.join(dir, 'test_predictions.npy'),  train_obj['test_predictions'])
    np.save(os.path.join(dir, 'test_scores.npy'),  train_obj['test_scores'])
    z_test.to_csv(os.path.join(dir, 'test_metadata.csv'))

    # Save the experiment configuration to JSON file
    json_config_file = os.path.join(dir, "train_config.json")
    with open(json_config_file, 'w') as f:
        json.dump(train_filter_profile, f)

    # Save the experiment configuration to JSON file
    json_config_file = os.path.join(dir, "test_config.json")
    with open(json_config_file, 'w') as f:
        json.dump(test_filter_profile, f)

    #############################################
    ###### Fit the conformal model
    ##############################################

    model = torch.load(os.path.join(dir, 'model.pt'), weights_only=False)
    cdataloader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val)), batch_size=args.batch_size, shuffle=False)
    cmodel = ConformalModel(model, cdataloader, alpha=args.alpha, lamda_criterion='size').to(device)
    torch.save(cmodel, os.path.join(dir, 'cmodel.pt'))

