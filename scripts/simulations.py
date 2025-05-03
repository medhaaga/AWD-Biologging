import sys
import os
import time
import torch
import json
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd
import config as config
import src.data_prep.create_segments_and_metadata as create_segments_and_metadata
import src.data_prep.create_matched_data_objects as create_matched_data_objects
                                                    
from src.utils.simulations import generate_acc_data
import src.utils.data as utils_data
import src.utils.io as utils_io
import src.utils.plot as utils_plot
from src.methods.prediction_model import create_dynamic_conv_model
from src.utils.train import train_run
##############################################
# Arguments
##############################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_name", type=str, default='class_imbalance', choices=['class_imbalance', 'temporal_context', 'distribution_shift'])
    parser.add_argument("--experiment_name", type=str, default='no_split', choices=['no_split', 'interdog', 'interyear', 'interAMPM'])
    parser.add_argument("--n_individuals", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5, help="size fo kernel for CNN")
    parser.add_argument("--n_channels", type=int, default=16, help="number of output channels for the first CNN layer")
    parser.add_argument("--n_CNNlayers", type=int, default=3, help="number of convolution layers")
    parser.add_argument("--window_duration_percentile", type=float, default=None, help="audio duration cutoff percentile")
    parser.add_argument("--window_duration", type=float, default=30.0, help="audio duration")
    parser.add_argument("--num_epochs", type=int, default=200)
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
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--match", type=int, default=0, help="should the matching be done or use pre-matched observations?")
    parser.add_argument("--min_duration", type=float, default=1.0, help="minimum duration of a behavior in seconds so that it is not discarded")
    parser.add_argument("--create_class_imbalance", type=int, default=0, help="whether to create class imbalance artificially")
    parser.add_argument("--class_imbalance_percent", type=float, default=0.01, help="percetage of feeding behavior in the imbalanced dataset")
    parser.add_argument("--alpha", type=float, default=0.05, help="coverage for RAPS is 1-alpha")
    parser.add_argument("--verbose", type=int, default=0)

    
    return parser

if __name__ == '__main__':

    # parse arguments
    parser = parse_arguments()
    args = parser.parse_args()

    individuals = [f"individual{i}" for i in range(1, args.n_individuals + 1)]
    individuals_years = {i: [2022] for i in individuals}
    behavior_prob = [0.05, 0.2, 0.4, 0.35]

    max_durations = {
        "Feeding": {"min": 2, "sec": 60},
        "Moving": {"min": 4, "sec": 60},
        "Resting": {"min": 6, "sec": 60},
        "Vigilant": {"min": 6, "sec": 60}
    }

    # Convert to DataFrame
    max_durations = pd.DataFrame.from_dict(max_durations, orient="index")
    max_durations.index.name = "Behavior"
    max_durations.columns.name = "DurationUnit"

    print("CLASS DISTRIBUTION")
    for b, p in zip(config.SIM_BEHAVIORS, behavior_prob):
        print(f"{b:<10}: {p}")

    # specify paths
    # Create directories and generate acceleration data
    os.makedirs(config.TEST_ROOT_DIR, exist_ok=True)
    results_dir = os.path.join(config.TEST_ROOT_DIR, args.sim_name)
    os.makedirs(results_dir, exist_ok=True)

    # define paths for various objects for this result subsection
    annotations_path = results_dir+"/test_all_annotations.csv"
    metadata_path = results_dir+"/test_metadata.csv"
    data_path = os.path.join(results_dir, "data")
    os.makedirs(data_path, exist_ok=True)
    training_results_dir = os.path.join(results_dir, "training_results")
    os.makedirs(training_results_dir, exist_ok=True)

    matched_summary_path = data_path+"/matched_acc_summary.csv"
    matched_data_path = data_path+"/matched_acc_data.csv"
    matched_metadata_path = data_path+"/matched_acc_metadata.csv"

    generate_acc_data(individuals_years, results_dir, behavior_prob, max_durations=max_durations, data_constants=config.SIMULATION_CONSTANTS)

    test_paths = {ind: f"{results_dir}/{ind}" for ind in individuals}

    # create halfday segments
    create_segments_and_metadata.run_vectronics(test_paths, max_chunks=5, verbose=True)

    # create metadata
    create_segments_and_metadata.create_metadata(test_paths, metadata_path)

    # load metadata and annotations
    metadata = pd.read_csv(metadata_path)
    all_annotations = pd.read_csv(annotations_path)

    acc_summary, acc_data, acc_data_metadata = create_matched_data_objects.create_matched_data(metadata, all_annotations)

    # save the matched data objects 
    acc_summary.to_csv(matched_summary_path)
    acc_data.to_csv(matched_data_path)
    acc_data_metadata.to_csv(matched_metadata_path)

    classes, counts = np.unique(acc_data.behavior, return_counts=True)
    proportions = counts / len(acc_data)
    print("CLASS PROPORTIONS")
    for cls, prop in zip(classes, proportions):
        print(f"{cls}: {prop:.3f}")

    start = time.time()
    X_train, y_train, z_train, X_val, y_val, z_val, X_test, y_test, z_test, _ = utils_data.setup_data_objects(metadata=metadata, 
                                                                                                        all_annotations=all_annotations, 
                                                                                                        collapse_behavior_mapping={}, 
                                                                                                        behaviors=config.SIM_BEHAVIORS, 
                                                                                                        args=args, 
                                                                                                        reuse_behaviors=config.SIM_BEHAVIORS,
                                                                                                        acc_data_path=matched_data_path,
                                                                                                        acc_metadata_path=matched_metadata_path
                                                                                                        )
    n_timesteps, n_features, n_outputs = X_train.shape[2], X_train.shape[1], len(np.unique(np.concatenate((y_train, y_val, y_test))))

    print("Train Class distribution")
    print("==========================")
    print(pd.DataFrame(np.unique(y_train, return_counts=True)[1]))
    print("")

    time_diff = time.time() - start

    print("")
    print(f'Creating data objects takes {time_diff:.2f} seconds.')
    print("")
    print("Shape of dataframes")
    print("==========================")
    print(f"{'Set':<8} {'X':<15} {'Y':<10} {'Z':<15}")
    print(f"{'Train:':<8} {str(X_train.shape):<15} {str(y_train.shape):<10} {str(z_train.shape):<15}")
    print(f"{'Val:':<8} {str(X_val.shape):<15} {str(y_val.shape):<10} {str(z_val.shape):<15}")
    print(f"{'Test:':<8} {str(X_test.shape):<15} {str(y_test.shape):<10} {str(z_test.shape):<15}")

    thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for theta in thetas:

        args.theta = theta

        #########################################
        #### Dataloaders
        #########################################

        train_dataloader, val_dataloader, test_dataloader = utils_data.setup_multilabel_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args)

        #########################################
        #### Model, loss, and optimizer
        #########################################

        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

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

        #########################################
        #### Training
        #########################################

        train_obj = train_run(model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader, args, device)
        model = train_obj['model']
        training_stats = train_obj['training_stats']

        dir = os.path.join(training_results_dir, f"theta{args.theta}")
        os.makedirs(dir, exist_ok=True)
        torch.save(model, os.path.join(dir, 'model.pt'))
        json_training_stats_file = os.path.join(dir, 'training_stats.json')
        with open(json_training_stats_file, 'w') as f:
            json.dump(training_stats, f)

        
        # save true and predicted test classes along with test metadata
        np.save(os.path.join(dir, 'test_true_classes.npy'), train_obj['test_true_classes'])
        np.save(os.path.join(dir, 'test_predictions.npy'),  train_obj['test_predictions'])
        np.save(os.path.join(dir, 'test_scores.npy'),  train_obj['test_scores'])

        # save true and predicted validation classes along with val metadata
        np.save(os.path.join(dir, 'val_true_classes.npy'),  train_obj['val_true_classes'])
        np.save(os.path.join(dir, 'val_predictions.npy'),  train_obj['val_predictions'])
        np.save(os.path.join(dir, 'val_scores.npy'),  train_obj['val_scores'])


                    
        