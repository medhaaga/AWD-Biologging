import sys
import os
import warnings
sys.path.append('../')
sys.path.append('../../')
import pandas as pd
import numpy as np
import torch



def online_score_evaluation(model_dir, acc_df, window_duration=None, window_length=None, hop_length=None, sampling_frequency=16, device='cpu'):

    required_cols = ['Timestamp', 'Acc X [g]', 'Acc Y [g]', 'Acc Z [g]']
    missing = [c for c in required_cols if c not in acc_df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (window_length is None) & (window_duration is None):
        raise ValueError('A window length/duration for the classification model is required.')
    
    if (window_length is None) & (window_duration is not None):
        window_length = int(window_duration*sampling_frequency)

    if (window_length is not None) & (window_duration is not None):
        assert window_length == int(window_duration*sampling_frequency), "window length and window duration are not compatible according to provided sampling frequency."

    if len(acc_df) < window_length:
        warnings.warn(f'Fewer samples than window length.')

    if hop_length is None:
        hop_length = window_length

    # check if model and window duration are compatible
    cmodel = torch.load(os.path.join(model_dir, 'cmodel.pt'), weights_only=False, map_location=device)

    zero_signal = torch.zeros(1, 3, window_length).to(device)
    assert cmodel.model[:-2](zero_signal).shape[-1] == cmodel.model[-2].in_features, "Window duration and model not compatible"

    windows, acc_segments = [], []
    start_index = 0

    # X = X[:, :, (X.shape[2] - window_length) % hop_length : ]

    while start_index + window_length < len(acc_df):
        end_index = start_index  + window_length
        window = acc_df.iloc[start_index:end_index]

        # Collect timestamps for start and end of the window
        window_start = window['Timestamp'].iloc[0]
        window_end = window['Timestamp'].iloc[-1]
        windows.append({'Timestamp start': window_start, 
                            'Timestamp end': window_end})
        
        # Collect values for tensor
        acc_segments.append(window[['Acc X [g]', 'Acc Y [g]', 'Acc Z [g]']].values)
        start_index += hop_length

    windows = pd.DataFrame(windows)

    # Convert list of arrays to a PyTorch tensor
    acc_segments = np.array(acc_segments).reshape(len(acc_segments), window_length, 3)
    acc_segments = np.transpose(acc_segments, (0,2,1)) # (number of windows, 3, window length)
    acc_segments = torch.tensor(acc_segments, dtype=torch.float32)
    with torch.no_grad():
        outputs, _ = cmodel(acc_segments.to(device))
    scores_np = np.array(outputs.unsqueeze(0).detach().cpu().numpy()) # (1, number of windows, number of classes)
    scores_np = np.transpose(scores_np, (0,2,1)) # (1, number of classes, number of windows)

    assert scores_np.shape[-1] == len(windows)

    return scores_np, windows

def online_smoothening(scores, start_times, window_len, hop_len):

    scores = scores.reshape(-1, scores.shape[-1]) #(number of classes, number of windows)

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    #  Validate that the number of timestamps matches the number of scores
    if len(start_times) != scores.shape[1]:
        raise ValueError("Length of start_times must match the number of scores.")

    n_windows = 1+ (scores.shape[-1] - window_len)//hop_len

    online_avg = np.zeros((scores.shape[0], n_windows))
    midpoint_times = np.zeros(n_windows, dtype='datetime64[ns]')

    for i in range(n_windows):
        start_idx = i * hop_len
        end_idx = start_idx + window_len

        online_avg[:,i] = np.mean(scores[:, start_idx:end_idx], axis=-1)
        # Get the start time of the first element and the last element in the window
        window_start_time = start_times[start_idx]
        window_end_time = start_times[end_idx - 1] 

        # Calculate the midpoint time of the window's time span
        midpoint_times[i] = midpoint_times[i] = window_start_time + (window_end_time - window_start_time) / 2
        
    return online_avg, midpoint_times
        

