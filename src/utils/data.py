import sys
sys.path.append('.')
sys.path.append('../../')

# Data
import numpy as np
import pandas as pd
import warnings
import time
import ast

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Torch
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

# Script imports
from src.data_prep.data_prep_utils import (filter_data, 
                                       apply_band_pass_filter)

from src.data_prep.create_matched_data_objects import (create_matched_data, 
                                                   create_padded_or_truncated_data)
from src.utils.io import (format_time,
                      get_matched_data_path,
                      get_matched_metadata_path)   

from config.settings import SAMPLING_RATE


def get_exp_filter_profiles(exp_name):

    """Return the train and test filter profiles for our different experiments."""

    train_filter_profile = {'individual ID': None,
                   'year': None,
                   'UTC Date [yyyy-mm-dd]': None,
                   'am/pm': None,
                   'half day [yyyy-mm-dd_am/pm]': None,
                   'avg temperature [C]': None}

    test_filter_profile = {'individual ID': None,
                'year': None,
                'UTC Date [yyyy-mm-dd]': None,
                'am/pm': None,
                'half day [yyyy-mm-dd_am/pm]': None,
                'avg temperature [C]': None}

    if exp_name == 'no_split':
        pass
        
    elif exp_name == 'interdog':
        train_filter_profile['individual ID'] = ['jessie', 'palus', 'ash', 'fossey']
        test_filter_profile['individual ID'] = ['green',]
    
    elif exp_name == 'interyear':
        train_filter_profile['year'] = [2021,]
        test_filter_profile['year'] = [2022,]
    
    elif exp_name == 'interAMPM':
        train_filter_profile['am/pm'] = ['am',]
        test_filter_profile['am/pm'] = ['pm',]

    elif exp_name == 'test_interyear':
        train_filter_profile['year'] = [2022,]
        test_filter_profile['year'] = [2025,]

    else:
        raise ValueError("Unspecified experiment name")

    return train_filter_profile, test_filter_profile


def train_test_metadata_split(train_metadata, test_metadata, test_size=0.2, random_state=0):
    
    """Create a split of train and test metedata so that there is no overlap. 
    Takes the train and test metadat, finds overlapping rows and divides them between 
    train and test data according to proportion of test data.

    Arguments
    --------------------
    train_metadata: pandas DataFrame
    test_metadata: pandas DataFrame
    test_size: float = 0.2

    Returns 
    ---------------------
    train_metadata: pandas DataFrame
    test_metadata: pandas DataFrame

    """
    
    # Perform inner join to extract overlapping rows
    overlapping_df = pd.merge(train_metadata, test_metadata, on=None, how='inner')

    # Perform left join to remove overlapping rows from df1
    df1_no_overlap = pd.merge(train_metadata, overlapping_df, on=None, how='left', indicator=True)
    df1_no_overlap = df1_no_overlap[df1_no_overlap['_merge'] == 'left_only']
    df1_no_overlap.drop(columns='_merge', inplace=True)

    # Perform right join to remove overlapping rows from df2
    df2_no_overlap = pd.merge(overlapping_df, test_metadata, on=None, how='right', indicator=True)
    df2_no_overlap = df2_no_overlap[df2_no_overlap['_merge'] == 'right_only']
    df2_no_overlap.drop(columns='_merge', inplace=True)

    overlap_train, overlap_test = train_test_split(overlapping_df, test_size=test_size, shuffle=True, random_state=random_state)

    df1 = pd.concat([df1_no_overlap, overlap_train])
    df2 = pd.concat([df2_no_overlap, overlap_test])

    return df1, df2

def split_overlapping_indices(train_indices, test_indices, behaviors, split_ratio=0.5):

    # Convert indices to sets for easier manipulation
    train_set = set(train_indices)
    test_set = set(test_indices)
    
    # Find overlapping indices
    overlapping_indices = train_set & test_set
    
    # Find non-overlapping indices
    train_non_overlap = train_set - overlapping_indices
    test_non_overlap = test_set - overlapping_indices
    
    # Convert sets back to sorted lists
    train_non_overlap = sorted(list(train_non_overlap))
    test_non_overlap = sorted(list(test_non_overlap))
    
    # Convert overlapping_indices to a sorted list
    overlapping_indices = np.array(sorted(list(overlapping_indices)))
    print(f'Overlapping indices of shape = {overlapping_indices.shape}')

    # train validation split 
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)

    for train_index, test_index in stratified_splitter.split(overlapping_indices, behaviors[overlapping_indices]):
        train_overlap_split, test_overlap_split = list(overlapping_indices[train_index]), list(overlapping_indices[test_index])
    
    # Combine non-overlapping indices with the split overlapping indices
    final_train_indices = train_non_overlap + train_overlap_split
    final_test_indices = test_non_overlap + test_overlap_split
    
    return final_train_indices, final_test_indices

###################################################
######### Set up data objects and loaders
###################################################

def give_balanced_weights(theta, y):
    n_classes = len(np.unique(y))
    weights = theta*np.ones(n_classes)/n_classes + (1-theta)*np.unique(y, return_counts=True)[1]/len(y)
    return weights

def adjust_behavior_and_durations(df, collapse_behavior_mapping, behaviors, verbose=False):

    """
        1. Collapse behaviors to coarser classes.
        2. filter out behaviors of interest
        3. remove behaviors of shorter duration
        4. remove running, moving, eatng, and marking behaviors shorter than 8 sec
    """

    if verbose:
        duration_before_filter = df.duration.sum()
        print(f'Total behavior duration before filtering - {duration_before_filter/3600}')  

    # collapse classes
    df['behavior'] = df['behavior'].replace(collapse_behavior_mapping) # collapse behaviors
    df = df[df['behavior'].isin(behaviors)]

    if verbose:
        duration_sum = df.duration.sum()
        print(f'Total duration after filtering out chosen behaviors is {duration_sum/3600} hrs.')

    # filtration
    df = df[df['duration'] >= 1]
    idx = [(df['behavior'].isin(['Running', 'Feeding', 'Moving'])) & (df['duration'] < 8) & (df['Source'] == 'Video')][0] # running from video lables of duration lesser than 8 sec are unreliable
    df = df[~idx]

    if verbose:
        duration_sum = df.duration.sum()
        print(f'Total behavior duration after filtering is {duration_sum/3600} hrs.')

    return df

def create_data_tensors(acc_data, collapse_behavior_mapping, behaviors, sampling_rate, 
                        padding='repeat', reuse_behaviors=[], min_duration=1.0, duration_percentile=50):
    """
    Create data tensors from acceleration data with given parameters.

    Parameters:
    - acc_data (pd.DataFrame): DataFrame containing acceleration data with 'acc_x', 'acc_y', 'acc_z' columns.
    - window_duration (float): Duration of the window in seconds for each data segment.
    - collapse_behavior_mapping (dict): Mapping to collapse behavior labels.
    - behaviors (list): List of behaviors to consider.
    - sampling_rate (float): Sampling rate of the acceleration data.
    - padding (str): Padding method, can be 'repeat' or other supported methods.
    - reuse_behaviors (list): List of behaviors to be reused via multiple winbdows extraction.
    - min_duration (float): Minimum duration of behavior to be considered, in seconds.

    Returns:
    - X (np.ndarray): Processed acceleration data tensor.
    - y (np.ndarray): Corresponding behavior labels tensor.
    - z (pd.DataFrame): Metadata for the data tensor.
    """

    # Convert stringified lists to actual lists
    acc_data['acc_x'] = acc_data['acc_x'].apply(ast.literal_eval)
    acc_data['acc_y'] = acc_data['acc_y'].apply(ast.literal_eval)
    acc_data['acc_z'] = acc_data['acc_z'].apply(ast.literal_eval)

    acc_data = adjust_behavior_and_durations(acc_data, collapse_behavior_mapping, behaviors)
    acc_data.reset_index(drop=True, inplace=True)
    window_duration = np.percentile(acc_data['duration'].values, duration_percentile)
    print(f'Duration of window is {window_duration} sec.')
    max_steps = int(window_duration * sampling_rate)

    X, y, z = create_padded_or_truncated_data(acc_data, max_steps, padding=padding, 
                                              reuse_behaviors=reuse_behaviors, min_duration=min_duration)
    return X, y, z

def match_train_test_df(metadata, all_annotations, collapse_behavior_mapping, behaviors, args):

    """match train and test data based on filter profiles and create train-test df
    
    Arguments:
    ----------------
    metadata: pd DataFrame
    all_annotations: pd DataFrame = all annotations from videos and audios
    collapse_behavior_mapping: dictionary 
    behaviors: list = list of desired behaviors
    args: dictionary
    
    Returns
    -----------------
    df_train: pd DataFrame
    df_test: pd DataFrame
    """

    # get filter profiles for this experiment
    train_filter_profile, test_filter_profile = get_exp_filter_profiles(args.experiment_name) 
    train_filtered_metadata = metadata.iloc[filter_data(metadata, train_filter_profile)]
    test_filtered_metadata = metadata.iloc[filter_data(metadata, test_filter_profile)]

    if len(pd.merge(train_filtered_metadata, test_filtered_metadata, on=None, how='inner')):
        warnings.warn("train and test filters overlap", UserWarning)
        print(f'Before overlap, \nno. of train half days: {len(train_filtered_metadata)}, no. of test half days: {len(test_filtered_metadata)}')
        train_filtered_metadata, test_filtered_metadata = train_test_metadata_split(train_filtered_metadata, test_filtered_metadata, args.train_test_split)
        print(f'After removing overlaps, \nno. of train half days: {len(train_filtered_metadata)}, no. of test half days: {len(test_filtered_metadata)}')
    else:
        print(f'No overlaps. \nno. of train half days: {len(train_filtered_metadata)}, no. of test half days: {len(test_filtered_metadata)}')

    t1 = time.time()

    # match filtered data with annotations
    _, df_train, _ = create_matched_data(train_filtered_metadata, all_annotations)
    _, df_test, _ = create_matched_data(test_filtered_metadata, all_annotations)

    t2 = time.time()

    print("")
    print("==================================")
    print(f'Data frames matched in {format_time(t2-t1)}.')

    df_train = adjust_behavior_and_durations(df_train, collapse_behavior_mapping, behaviors)
    df_test = adjust_behavior_and_durations(df_test, collapse_behavior_mapping, behaviors)

    df_train.reset_index()
    df_test.reset_index()

    return df_train, df_test

def load_matched_train_test_df(collapse_behavior_mapping, behaviors, exp_name, acc_data_path, acc_metadata_path, train_test_split=0.2):


    """load pre-matched accereation-behavior data. Create train and test dataframes based on filter profiles
    
    Arguments:
    ----------------
    collapse_behavior_mapping: dictionary 
    behaviors: list = list of desired behaviors
    args.experiment_name: string = name of experiment provided to `get_exp_filter_profiles` function.
    acc_data_path: string = path where matched acceleration data is stored
    acc_metadata_path: string = path where matched acceleration metadata is stored
    train_test_split: float = proportion of test data in overlapping data points
    
    Returns
    -----------------
    df_train: pd DataFrame
    df_test: pd DataFrame
    """

    train_filter_profile, test_filter_profile = get_exp_filter_profiles(exp_name) 
    
    acc_data = pd.read_csv(acc_data_path)
    acc_data_metadata = pd.read_csv(acc_metadata_path)

    acc_data['acc_x'] = acc_data['acc_x'].apply(ast.literal_eval)
    acc_data['acc_y'] = acc_data['acc_y'].apply(ast.literal_eval)
    acc_data['acc_z'] = acc_data['acc_z'].apply(ast.literal_eval)

    acc_data = adjust_behavior_and_durations(acc_data, collapse_behavior_mapping, behaviors)
    acc_data_metadata = acc_data_metadata.loc[acc_data.index]

    acc_data.reset_index()
    acc_data_metadata.reset_index()

    print(f'Total number of matched annotations: {len(acc_data)}')

    train_filter_idx = filter_data(acc_data_metadata, train_filter_profile)
    test_filter_idx = filter_data(acc_data_metadata, test_filter_profile)


    if len(set(train_filter_idx) & set(test_filter_idx)):
        warnings.warn("train and test filters overlap", UserWarning)
        print(f'Before overlap, \nno. of train observations: {len(train_filter_idx)}, no. of test observations: {len(test_filter_idx)}')
        train_filter_idx, test_filter_idx = split_overlapping_indices(train_filter_idx, test_filter_idx, acc_data['behavior'].values, split_ratio=train_test_split)
        print(f'After removing overlaps, \nno. of train observations: {len(train_filter_idx)}, no. of test observations: {len(test_filter_idx)}')
    else:
        print(f'No overlaps. \nno. of train observations: {len(train_filter_idx)}, no. of test observations: {len(test_filter_idx)}')


    df_train = acc_data.iloc[train_filter_idx]
    df_test = acc_data.iloc[test_filter_idx]

    df_train.reset_index()
    df_test.reset_index()

    return df_train, df_test


def setup_data_objects(metadata, all_annotations, collapse_behavior_mapping, 
                        behaviors, args, reuse_behaviors=[], acc_data_path=None, 
                        acc_metadata_path=None):

    """
    Arguments
    -----------------------
    metadata: Pandas Dataframe = metadata on all acceleration segments
    all_annotations: Pandas Dataframe = information on data frames
    collapse_behavior_mapping: dictionary 
    behaviors: list = list of behaviors of interest
    args: dictionary 
    match: bool = whether to match behaviors or use a pre-matched dataframe
    acc_data_path: string = path where matched acceleration data is stored, default=None
    acc_metadata_path: string = path where matched acceleration metadata is stored, default=None
    

    Returns 
    ----------------------
    X_train     : (n, d, T) np ndarray = train acceleration, n = no. of samples, d = no. of features, T = time axis            
    y_train     : (n, ) np ndarray    = train labels, n = no. of samples
    z_train     : pandas dataframe     = metadata associated with the train observations                                       
    X_val       : (n, d, T) np ndarray = val acceleration, n = no. of samples, d = no. of features, T = time axis           
    y_val       : (n, ) np ndarray    = val labels, n = no. of samples
    z_val       : pandas dataframe     = metadata associated with the validation observations                                  
    X_test      : (n, d, T) np ndarray = test acceleration, n = no. of samples, d = no. of features, T = time axis             
    y_test      : (n, ) np ndarray    = test labels, n = no. of samples
    z_test      : pandas dataframe     = metadata associated with the test observations                                        
    """

    t1 = time.time()
    if args.match or (acc_data_path is None) or (acc_metadata_path is None):
        print('Matching acceleration-behavior pairs...')
        df_train, df_test = match_train_test_df(metadata, all_annotations, collapse_behavior_mapping, behaviors, args)
    else:
        print('Using pre-matched acceleration-behavior pairs...')
        df_train, df_test = load_matched_train_test_df(collapse_behavior_mapping=collapse_behavior_mapping, 
                                                        behaviors=behaviors, 
                                                        exp_name=args.experiment_name, 
                                                        acc_data_path=acc_data_path,
                                                        acc_metadata_path=acc_metadata_path,
                                                        train_test_split=args.train_test_split)

    print("")
    print("==================================")
    print(f"Matching annotations to acceleration snippets takes {time.time() - t1:3f} seconds")

    t2 = time.time()
    # fix sequence max length and truncate/pad data to create X, y, and z.
    max_acc_duration = np.percentile(np.concatenate((df_train['duration'].values, df_test['duration'].values), axis=0), args.window_duration_percentile)
    max_steps = int(max_acc_duration*SAMPLING_RATE)
    X, y, z = create_padded_or_truncated_data(df_train, max_steps, padding=args.padding, reuse_behaviors=reuse_behaviors, min_duration=args.min_duration)
    X_test, y_test, z_test = create_padded_or_truncated_data(df_test, max_steps, padding=args.padding, reuse_behaviors=reuse_behaviors, min_duration=args.min_duration)
    print(f"Creating fixed-duration windows takes {time.time() - t2:3f} seconds.")

    print("")
    print("==================================")
    print(f"Time series duration window = {max_acc_duration}")

    # Band filter - no filter by default
    X = apply_band_pass_filter(X, args.cutoff_frequency, SAMPLING_RATE, btype=args.filter_type, N=args.cutoff_order, axis=2)
    X_test = apply_band_pass_filter(X_test, args.cutoff_frequency, SAMPLING_RATE, btype=args.filter_type, N=args.cutoff_order, axis=2)

    # standardize data
    if args.normalization:
        X = (X - np.mean(X, axis=0, keepdims=True))/np.std(X, axis=0, keepdims=True)
        X_test = (X_test - np.mean(X_test, axis=0, keepdims=True))/np.std(X_test, axis=0, keepdims=True)

    # label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate((y, y_test)))
    y = label_encoder.transform(y)
    y_test = label_encoder.transform(y_test)
    
    # train validation split 
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.train_val_split, random_state=42)

    for train_index, val_index in stratified_splitter.split(X, y):
        X_train, X_val, y_train, y_val, z_train, z_val = X[train_index], X[val_index], y[train_index], y[val_index], z.iloc[train_index], z.iloc[val_index]

    return X_train, y_train, z_train, X_val, y_val, z_val, X_test, y_test, z_test, label_encoder



def setup_multilabel_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, args):

    n_outputs = len(np.unique(np.concatenate((y_train, y_val, y_test))))
    
    weights = give_balanced_weights(args.theta, y_train)
    y_weights = torch.tensor([weights[i] for i in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(y_weights, len(y_weights))

    # converting to one-hot vectors
    y_train = np.eye(n_outputs)[y_train]
    y_val = np.eye(n_outputs)[y_val]
    y_test = np.eye(n_outputs)[y_test]

    # Convert data and labels to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoader for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def create_artificial_class_imbalance(X, y, percent):
    feeding_indices = np.where(y == 0)[0]
    np.random.shuffle(feeding_indices)
    keep_count = int(percent * len(y))
    print(f'Keeping {keep_count} many feeding behaviors.')
    keep_indices = feeding_indices[:keep_count]

    # Create a mask to keep the selected indices
    mask = np.ones(len(y), dtype=bool)
    mask[feeding_indices] = False
    mask[keep_indices] = True
    return X[mask], y[mask]