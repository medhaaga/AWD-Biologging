# System & OS
import os
import sys
import argparse
import warnings
import random as random
sys.path.append('.')
sys.path.append('../')

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder

# Script imports

from src.methods.online_predictions import (online_score_evaluation,
                                            online_smoothening)

from config.settings import (SAMPLING_RATE,
                             AWD_VECTRONICS_PATHS,
                             VECTRONICS_METADATA_PATH,
                             BEHAVIORS,
                             VECTRONICS_BEHAVIOR_EVAL_PATH,
                             COLLAPSE_BEHAVIORS_MAPPING)

from src.utils.io import (get_results_path,
                          get_online_pred_path,
                          get_matched_data_path)

from src.utils.data import (adjust_behavior_and_durations)

from src.utils.plot import (plot_signal_and_online_predictions)

# for reproducible results, conduct online evaluations for dog jessie with seed 23. This is 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_duration", type=float, default=12.937)
    parser.add_argument("--window_duration_percentile", type=float, default=50)
    parser.add_argument("--window_length", type=int, default=206)
    parser.add_argument("--score_hop_length", type=int, default=None)
    parser.add_argument("--smoothening_window_length", type=int, default=10)
    parser.add_argument("--smoothening_hop_length", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default='no_split', choices=['no_split', 'interdog', 'interyear', 'interAMPM'])
    parser.add_argument("--kernel_size", type=int, default=5, help="size fo kernel for CNN")
    parser.add_argument("--n_channels", type=int, default=32, help="number of output channels for the first CNN layer")
    parser.add_argument("--n_CNNlayers", type=int, default=5, help="number of convolution layers")
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dog", type=str, default='jessie', choices=['jessie', 'ash', 'palus', 'green', 'fossey'])


    return parser


def halfday_online_eval(csv_path, model_dir, window_duration, window_length, smoothening_config):

    acc_data = pd.read_csv(csv_path)
    acc_data['Timestamp'] = pd.to_datetime(acc_data['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    acc_data.sort_values(by='Timestamp', inplace=True)
    signal = torch.tensor(np.array([acc_data['Acc X [g]'].values, acc_data['Acc Y [g]'].values, acc_data['Acc Z [g]'].values])).float().unsqueeze(0)
    scores = online_score_evaluation(model_dir=model_dir, X=signal, window_duration=window_duration, window_length=window_length, hop_length=smoothening_config['score_hop_length'], sampling_frequency=SAMPLING_RATE)
    online_avg = online_smoothening(scores, smoothening_config['smoothening_window_length'], smoothening_config['smoothening_hop_length'])

    return acc_data['Timestamp'].values, signal, scores, online_avg
    

def random_halfday_online_eval(model_config, dog, window_duration, window_length, smoothening_config, save_objects=True, plot=True):
    """Evaluates a model on a randomly selected half-day accelerometer file.
    
    Conducts smoothening for online eval, and optionally saves and plots the results.

    Parameters:
    --------------------------
    - model_config: dict = Configuration for the model.
    - dog: str = Identifier for the dog (used to locate accelerometer data).
    - window_duration: float = Duration of the window used in model training.
    - smoothening_config: dict = Configuration for smoothening including window length and hop length.
    - save_objects: bool = Whether to save the signal, scores, and online average arrays to files. Default is True.
    - plot: bool = Whether to plot the signal and online predictions. Default is True.

    Returns:
    ---------------------------
    - signal: numpy.ndarray = The raw signal data.
    - scores: numpy.ndarray = The scores computed by the model.
    - online_avg: numpy.ndarray = The online average prediction probabilities.
    """

    model_dir = get_results_path(
        exp_name=model_config['experiment_name'], 
        n_CNNlayers=model_config['n_CNNlayers'], 
        n_channels=model_config['n_channels'], 
        kernel_size=model_config['kernel_size'], 
        theta=model_config['theta'],
        window_duration_percentile=model_config['window_duration_percentile']
    )
    acc_dir = os.path.join(AWD_VECTRONICS_PATHS[dog], 'combined_acc')
    matched_acc_data = pd.read_csv(get_matched_data_path())
    matched_acc_data = adjust_behavior_and_durations(matched_acc_data, COLLAPSE_BEHAVIORS_MAPPING, BEHAVIORS)

    half_day = random.choice(matched_acc_data[(matched_acc_data['dog ID'] == dog)]['half day [yyyy-mm-dd_am/pm]'].values)
    acc_file_path = os.path.join(acc_dir, dog + '_' + half_day + '.csv')

    half_day_behaviors = matched_acc_data[(matched_acc_data['dog ID'] == dog) & (matched_acc_data['half day [yyyy-mm-dd_am/pm]'] == half_day)]
    half_day_behaviors.loc[:, 'behavior_start'] = pd.to_datetime(half_day_behaviors['behavior_start'])
    half_day_behaviors.loc[:, 'behavior_end'] = pd.to_datetime(half_day_behaviors['behavior_end'])

    # Perform half-day online evaluation
    time, signal, scores, online_avg = halfday_online_eval(acc_file_path, model_dir, window_duration, window_length, smoothening_config)

    # Get the directory to save the online predictions
    save_dir = get_online_pred_path(os.path.basename(acc_file_path).split('.')[0])
    smoothening_window_len = smoothening_config['smoothening_window_length']
    save_dir = os.path.join(save_dir, f'window_length_{smoothening_window_len}')
    os.makedirs(save_dir, exist_ok=True)

    if save_objects:
        np.save(os.path.join(save_dir, 'signal.npy'), signal)
        np.save(os.path.join(save_dir, 'scores.npy'), scores)
        np.save(os.path.join(save_dir, 'online_avg.npy'), online_avg)


    if plot:

        label_encoder = LabelEncoder()
        label_encoder.fit(BEHAVIORS)

        plot_signal_and_online_predictions(
            time,
            signal, 
            online_avg, 
            smoothening_config['smoothening_window_length'], 
            smoothening_config['smoothening_hop_length'], 
            window_duration, 
            label_encoder, 
            sampling_rate=SAMPLING_RATE, 
            plot_dir=save_dir,
            half_day_behaviors=half_day_behaviors
        )

    return signal, scores, online_avg

def all_online_eval(model_config, device, sampling_frequency=16, window_length=None, window_duration=None):

    if (window_length is None) & (window_duration is None):
        raise ValueError('A window length/duratioon for the classification model is required.')
    
    if (window_length is None) & (window_duration is not None):
        window_length = int(window_duration*sampling_frequency)

    if (window_length is not None) & (window_duration is not None):
        assert window_length == int(window_duration*sampling_frequency), "window length and window duration are not compatible according to provided sampling frequency."
    
    # load the conformal model
    model_dir = get_results_path(
        model_config['experiment_name'], 
        model_config['n_CNNlayers'], 
        model_config['n_channels'], 
        model_config['kernel_size'], 
        model_config['theta']
    )
    # check if model and window duration are compatible
    cmodel = torch.load(os.path.join(model_dir, 'cmodel.pt')).to(device)
    zero_signal = torch.zeros(1, 3, window_length).to(device)
    assert cmodel.model[:-2](zero_signal).shape[-1] == cmodel.model[-2].in_features, "Window duration and model not compatible"

    # load metadata for access to all half days    
    metadata = pd.read_csv(VECTRONICS_METADATA_PATH) 

    # fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(BEHAVIORS)

    for _, row in tqdm(metadata.iterrows(), total = len(metadata)):

        dog, half_day = row['dog ID'], row['half day [yyyy-mm-dd_am/pm]']

        windows = []
        half_day_acc = []

        half_day_data = pd.read_csv(row['file path'])
        half_day_data['Timestamp'] = pd.to_datetime(half_day_data['Timestamp'], utc=True)

        if len(half_day_data) < window_length:
            warnings.warn(f'half day {dog}-{half_day} has lesser data than window length. Skipped.')
        
        start_index = 0

        while start_index + window_length < len(half_day_data):

            end_index = start_index  + window_length
            window = half_day_data.iloc[start_index:end_index]

            # Collect timestamps for start and end of the window
            window_start = window['Timestamp'].iloc[0]
            window_end = window['Timestamp'].iloc[-1]
            windows.append({'Timestamp start': window_start, 'Timestamp end': window_end})

            # Collect values for tensor
            window_values = window[['Acc X [g]', 'Acc Y [g]', 'Acc Z [g]']].values
            half_day_acc.append(window_values)

            start_index = end_index

        # Create DataFrame for windows
        half_day_online_evals = pd.DataFrame(windows)
        half_day_online_evals['Dog ID'] = [dog]*len(half_day_online_evals)

        # Convert list of arrays to a PyTorch tensor
        half_day_acc = np.array(half_day_acc).reshape(len(half_day_acc), window_length, 3)
        half_day_acc = np.transpose(half_day_acc, (0,2,1))
        half_day_acc = torch.tensor(half_day_acc, dtype=torch.float32)

        with torch.no_grad():
            scores, sets = cmodel(half_day_acc.to(device))

        half_day_online_evals['Prediction sets'] = [label_encoder.inverse_transform(ws) for ws in sets]
        half_day_online_evals['Prediction scores'] = [score.cpu().numpy()[ws] for (score, ws) in zip(scores, sets)]
        half_day_online_evals['Most probable behavior'] = label_encoder.inverse_transform(np.argmax(scores.cpu().numpy(), axis=1))
        half_day_online_evals['Eating'] = half_day_online_evals['Prediction sets'].apply(lambda behaviors: 'Eating' in behaviors).astype('int')
        half_day_online_evals['Running'] = half_day_online_evals['Prediction sets'].apply(lambda behaviors: 'Running' in behaviors).astype('int')


        save_dir = os.path.join(VECTRONICS_BEHAVIOR_EVAL_PATH, os.path.basename(row['file path']))
        half_day_online_evals.to_csv(save_dir, index=False)
    

def extract_running_events():

    all_files = os.listdir(VECTRONICS_BEHAVIOR_EVAL_PATH)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    cols = pd.read_csv(os.path.join(VECTRONICS_BEHAVIOR_EVAL_PATH, csv_files[0])).columns
    running_events = pd.DataFrame(columns=cols)

    for _,file_path in enumerate(csv_files):

        half_day_online_evals = pd.read_csv(os.path.join(VECTRONICS_BEHAVIOR_EVAL_PATH, file_path))
        df_temp = half_day_online_evals[half_day_online_evals['Running'] == 1]
        running_events = pd.concat([running_events, df_temp], ignore_index=True)

    running_events.to_csv(os.path.join(os.path.dirname(VECTRONICS_BEHAVIOR_EVAL_PATH), 'running_events.csv'), index=True)


def extract_eating_events():

    all_files = os.listdir(VECTRONICS_BEHAVIOR_EVAL_PATH)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    cols = pd.read_csv(os.path.join(VECTRONICS_BEHAVIOR_EVAL_PATH, csv_files[0])).columns
    running_events = pd.DataFrame(columns=cols)

    for _, file_path in enumerate(csv_files):

        half_day_online_evals = pd.read_csv(os.path.join(VECTRONICS_BEHAVIOR_EVAL_PATH, file_path))
        df_temp = half_day_online_evals[half_day_online_evals['Eating'] == 1]
        running_events = pd.concat([running_events, df_temp], ignore_index=True)

    running_events.to_csv(os.path.join(os.path.dirname(VECTRONICS_BEHAVIOR_EVAL_PATH), 'eating_events.csv'), index=True)



if __name__ == '__main__':

    parser = parse_arguments()
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_config = {'experiment_name': args.experiment_name,
                    'n_CNNlayers': args.n_CNNlayers,
                    'n_channels': args.n_channels,
                    'kernel_size': args.kernel_size,
                    'theta': args.theta,
                    'window_duration_percentile': args.window_duration_percentile
                    }

    smoothening_config = {'smoothening_window_length': args.smoothening_window_length,
                          'smoothening_hop_length': args.smoothening_hop_length,
                          'score_hop_length': args.score_hop_length
                          }

    random_halfday_online_eval(model_config=model_config, 
                               dog=args.dog, 
                               window_duration=args.window_duration, 
                               window_length=args.window_length, 
                               smoothening_config=smoothening_config, 
                               save_objects=True, 
                               plot=True)

    # all_online_eval(model_config, device, sampling_frequency=SAMPLING_RATE, window_length=None, window_duration=args.window_duration)
    # extract_running_events()
    # extract_eating_events()



    

