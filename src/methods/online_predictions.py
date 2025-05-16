import sys
import os

sys.path.append('../')
sys.path.append('../../')

import numpy as np
import torch



def online_score_evaluation(model_dir, X, window_duration=None, window_length=None, hop_length=None, sampling_frequency=16, device='cpu'):

    assert len(X.shape) == 3, "Signal is not 3-dimensional"

    if (window_length is None) & (window_duration is None):
        raise ValueError('A window length/duration for the classification model is required.')
    
    if (window_length is None) & (window_duration is not None):
        window_length = int(window_duration*sampling_frequency)

    if (window_length is not None) & (window_duration is not None):
        assert window_length == int(window_duration*sampling_frequency), "window length and window duration are not compatible according to provided sampling frequency."
    
    assert X.shape[2] >= window_length, "Signal is shorter than window size"

    if hop_length is None:
        hop_length = window_length

    # check if model and window duration are compatible
    cmodel = torch.load(os.path.join(model_dir, 'cmodel.pt'), weights_only=False).to(device)

    zero_signal = torch.zeros(1, 3, window_length).to(device)
    assert cmodel.model[:-2](zero_signal).shape[-1] == cmodel.model[-2].in_features, "Window duration and model not compatible"

    scores = []
    X = X[:, :, (X.shape[2] - window_length) % hop_length : ]


    for i in range(2 + (X.shape[2] - window_length)//hop_length):
        X_temp = X[:, :, i*hop_length : i*hop_length+window_length]
        
        with torch.no_grad():
            outputs, _ = cmodel(X_temp.to(device))
        
        scores.append(outputs.float().cpu().numpy())

    scores = np.array(scores).transpose(1, 2, 0) # (number of samples, number of windows, number of classes)

    return scores

def online_smoothening(scores, window_len, hop_len):

    scores = scores.reshape(-1,scores.shape[-1])
    n_windows = 1+ (scores.shape[-1] - window_len)//hop_len

    online_avg = np.zeros((scores.shape[0], n_windows))

    for i in range(n_windows):
        start = i*hop_len
        online_avg[:,i] = np.mean(scores[:, start:start+window_len], axis=-1)
        
    return online_avg

