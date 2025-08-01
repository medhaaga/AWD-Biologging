# import libraries
import sys
import os
import warnings
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
from scipy.stats import linregress
from scipy.stats import circvar
from scipy.spatial.distance import cdist
import seaborn as sns
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')

# import scripts

import config as config

# Graphing Parameters
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 30
mpl.rcParams['ytick.labelsize'] = 30
mpl.rcParams["axes.labelsize"] = 30
mpl.rcParams['legend.fontsize'] = 30
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'

np.random.seed(42)

def simulate_fourier_signal(frequencies, amplitudes, phases, n_samples, fs):
    """
    Simulates a signal based on a Fourier breakdown.
    
    Parameters:
    - frequencies: list of frequencies in Hz
    - amplitudes: list of amplitudes
    - phases: list of phase shifts in radians
    - start_time: datetime object
    - end_time: datetime object
    - fs: sampling frequency in Hz
    
    Returns:
    - pd.DataFrame with 'timestamp' and 'signal' columns
    """
    # Check that all input lists are the same length
    if not (len(frequencies) == len(amplitudes) == len(phases)):
        raise ValueError("frequencies, amplitudes, and phases must have the same length")
    
    # Generate time vector

    t = np.linspace(0, n_samples/fs, n_samples, endpoint=False)
    
    # Build signal from the Fourier components
    signal = np.zeros_like(t)
    for f, A, phi in zip(frequencies, amplitudes, phases):
        signal += A * np.sin(2 * np.pi * f * t + phi)
    
    
    return signal

def simulate_axis_signal(f, A, phi, sigma, n_samples, tau=0.1):

    f = [freq + np.random.uniform(-tau, tau) for freq in list(f)]
    A = [amp + np.random.uniform(-tau, tau) for amp in list(A)]
    phi = [ph + np.random.uniform(-tau, tau) for ph in list(phi)]
    sigma = sigma + np.random.uniform(-tau, tau, size=n_samples)

    signal = simulate_fourier_signal(f, A, phi, n_samples, config.SAMPLING_RATE)
    signal += np.random.normal(loc=0, scale=sigma)

    return signal

def generate_acc_data(individuals, results_dir, behavior_prob, max_durations, data_constants, n_behaviors_range=(300, 400)):
    """
    Simulates and saves synthetic accelerometry data and behavior annotations for multiple individuals over specified years.

    Parameters
    ----------
    individuals : dict
        A dictionary where keys are individual IDs (strings) and values are lists of years (ints) for which to generate data.
        Example: {'Individual1': [2021, 2022], 'Individual2': [2022]}

    results_dir : str
        Path to the output directory where per-individual folders and CSV files will be saved.

    behavior_prob : list of float
        A probability distribution over behaviors (same length and order as `config.SIM_BEHAVIORS`) used to randomly sample behaviors during the simulation.

    max_durations: pd.DataFrame
        A DataFrame with columns ['Behavior', 'min', 'sec', where:
            - the min and sec column give the max durstaion in minutes and seconds for each behavior

    data_constants : pd.DataFrame
        A DataFrame with columns ['Behavior', 'Axis', 'f', 'A', 'phi'], where:
            - 'Behavior' is a string (e.g., "Running")
            - 'Axis' is one of "X", "Y", or "Z"
            - 'f', 'A', and 'phi' are lists of floats representing frequencies, amplitudes, and phase shifts for a Fourier signal model of that behavior-axis.

    Returns
    -------
    None
        The function saves two types of files:
        1. Simulated accelerometry CSVs for each individual and year in `results_dir`.
        2. A single annotations CSV at `config.TEST_ANNOTATIONS_PATH` containing all generated behavior labels and their time intervals.
    """

    behavior_data = []

    # generate random accelerometry data for i day
    for individual, years in tqdm(individuals.items()):
        
        individual_dir = os.path.join(results_dir, individual)
        os.makedirs(individual_dir, exist_ok=True)
        
        for year in years:
            start_time = datetime(year, 1, 1)
            end_time = datetime(year, 1, 1, 23, 59, 59)
            current_time = start_time
            time_delta = timedelta(seconds=1/config.SAMPLING_RATE)
            
            timestamps = []
            while current_time <= end_time:
                timestamps.append(current_time)
                current_time += time_delta

            data = {
                "UTC Date[mm/dd]": [t.strftime("%m/%d") for t in timestamps],
                "UTC DateTime": [t.strftime("%H:%M:%S") for t in timestamps],
                "Milliseconds": [t.microsecond // 1000 for t in timestamps],
                "Acc X [g]": np.zeros(len(timestamps)),
                "Acc Y [g]": np.zeros(len(timestamps)),
                "Acc Z [g]": np.zeros(len(timestamps)),
                "Temperature [Celsius]": np.random.uniform(20, 30, len(timestamps))
            }
            
            acc_df = pd.DataFrame(data)

            num_behaviors = np.random.randint(*n_behaviors_range)
            existing_intervals = []

            attempts = 0
            max_attempts = 1000  # prevents infinite loop

            while len(existing_intervals) < num_behaviors and attempts < max_attempts:

                observed_behavior = np.random.choice(a=config.SIM_BEHAVIORS,
                                                        p=behavior_prob)

                attempts += 1
                behavior_start = start_time + timedelta(hours=np.random.randint(0, 24), 
                                                        minutes=np.random.randint(0, 60),
                                                        seconds=np.random.randint(5, 120))
                behavior_end = behavior_start + timedelta(minutes=np.random.randint(0, max_durations.loc[observed_behavior, 'min']),
                                                            seconds=np.random.randint(5, max_durations.loc[observed_behavior, 'sec']))

                if behavior_end > end_time:
                    behavior_end = end_time

                overlap = any(
                    not (behavior_end <= existing_start or behavior_start >= existing_end)
                    for (existing_start, existing_end) in existing_intervals
                )
                if overlap:
                    continue  # Skip and try again
                
                existing_intervals.append((behavior_start, behavior_end))
                
                
                behavior_data.append([individual, 
                                    observed_behavior, 
                                    behavior_start.strftime("%Y/%m/%d %H:%M:%S"),
                                    behavior_end.strftime("%Y/%m/%d %H:%M:%S"),
                                    "Video"])
                
                # Modify the acceleration values based on behavior

                df_full_timestamp = pd.to_datetime(str(year) + "/" + acc_df["UTC Date[mm/dd]"] + " " + acc_df["UTC DateTime"], format="%Y/%m/%d %H:%M:%S")
                mask = (df_full_timestamp >= pd.to_datetime(behavior_start, format="%Y/%m/%d %H:%M:%S")) & \
                    (df_full_timestamp <= pd.to_datetime(behavior_end, format="%Y/%m/%d %H:%M:%S"))
                n_samples = acc_df.loc[mask, "Acc X [g]"].shape[0]

                wrong_portion = np.random.uniform(0, 0.5)
                n_wrong = int(wrong_portion * n_samples)

                # Wrong behavior parameters
                if np.random.rand() < 0.5:
                    wrong_behavior = config.WRONG_BEHAVIORS[observed_behavior]
                else:
                    wrong_behavior = observed_behavior

                def get_signal(axis, behavior, n):
                    f, A, phi, sigma = data_constants.loc[
                        (data_constants["Behavior"] == behavior) & 
                        (data_constants["Axis"] == axis), ['f', 'A', 'phi', 'sigma']
                    ].values[0]
                    
                    return simulate_axis_signal(f, A, phi, sigma, n)
                
                true_indices = mask[mask].index
                selected_indices = true_indices[:n_wrong]
                acc_df.loc[selected_indices, "Acc X [g]"] += get_signal('X', wrong_behavior, n_wrong) 
                acc_df.loc[selected_indices, "Acc Y [g]"] += get_signal('Y', wrong_behavior, n_wrong) 
                acc_df.loc[selected_indices, "Acc Z [g]"] += get_signal('Z', wrong_behavior, n_wrong) 
                
                selected_indices = true_indices[n_wrong:]
                acc_df.loc[selected_indices, "Acc X [g]"] += get_signal('X', observed_behavior, n_samples-n_wrong) 
                acc_df.loc[selected_indices, "Acc Y [g]"] += get_signal('Y', observed_behavior, n_samples-n_wrong) 
                acc_df.loc[selected_indices, "Acc Z [g]"] += get_signal('Z', observed_behavior, n_samples-n_wrong) 

            acc_df.to_csv(os.path.join(individual_dir, f"{year}.csv"), index=False)

    # check if annottaions exist already

    annotations_df = pd.DataFrame(behavior_data, 
                                    columns=["id", "Behavior", "Timestamp_start", "Timestamp_end", "Source"])
    annotations_path = os.path.join(results_dir, "test_all_annotations.csv")

    # if os.path.exists(annotations_path):
    #     existing_df = pd.read_csv(annotations_path)
    #     annotations_df = pd.concat([existing_df, annotations_df], ignore_index=True)
        
    annotations_df.to_csv(annotations_path, index=False)


def simulate_markov_acc_day(data_constants, transition_matrix, avg_durations, tau=0.1):
    
    """
    Simulate a half-day of tri-axial acceleration data using a Markov process 
    to determine behavior transitions.

    Parameters
    ----------
    data_constants : pd.DataFrame
        A DataFrame containing the Fourier signal parameters (`f`, `A`, `phi`)
        for each behavior and axis ('X', 'Y', 'Z'). It must have columns:
        'Behavior', 'Axis', 'f', 'A', 'phi'.
    
    transition_matrix : np.ndarray
        A square Markov transition matrix of shape (B, B), where B is the number behaviors.

    avg_durations : pd.DataFrame
        A DataFrame with average durations for each behavior. It must have the behavior as index 
        and columns ['min', 'sec'] specifying the mean duration in minutes and seconds.

    Returns
    -------
    acc_df : pd.DataFrame
        A DataFrame containing the simulated accelerometer data with columns:
        ['Timestamp', 'Acc X [g]', 'Acc Y [g]', 'Acc Z [g]', 'Temperature [Celsius]', 'Behavior'].

    annotations_df : pd.DataFrame
        A DataFrame with behavior annotations corresponding to segments of the 
        generated signal. Columns include:
        ['id', 'Behavior', 'Timestamp_start', 'Timestamp_end', 'Source'].
    """
    
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    end_time = datetime(2023, 1, 1, 23, 59, 59)
    current_time = start_time

    behaviors = config.SIM_BEHAVIORS
    behavior_to_index = {b: i for i, b in enumerate(behaviors)}
    index_to_behavior = {i: b for b, i in behavior_to_index.items()}
    
    current_behavior = np.random.choice(behaviors)
    acc_data = []
    behavior_annotations = []

    while current_time < end_time:
        # Sample duration (e.g., 30–300 seconds)
        
        duration = timedelta(minutes=np.random.normal(loc=avg_durations.loc[current_behavior, 'min'], scale=1, size=1).item(),
                             seconds=np.random.normal(loc=avg_durations.loc[current_behavior, 'sec'], scale=1, size=1).item())
        segment_end = min(current_time + duration, end_time)
        n_samples = int((segment_end - current_time).total_seconds() * config.SAMPLING_RATE)

        timestamps = [current_time + timedelta(seconds=i/config.SAMPLING_RATE) for i in range(n_samples)]

        # Retrieve Fourier params for current behavior
        def get_signal(axis):
            f, A, phi, sigma = data_constants.loc[
                (data_constants["Behavior"] == current_behavior) & 
                (data_constants["Axis"] == axis), ['f', 'A', 'phi', 'sigma']
            ].values[0]
            
            return simulate_axis_signal(f, A, phi, sigma, n_samples, tau=tau)
        
        acc_x = get_signal("X") 
        acc_y = get_signal("Y") 
        acc_z = get_signal("Z") 

        for i in range(n_samples):
            acc_data.append({
                "Timestamp": timestamps[i],
                "Acc X [g]": acc_x[i],
                "Acc Y [g]": acc_y[i],
                "Acc Z [g]": acc_z[i],
                "Temperature [Celsius]": np.random.uniform(20, 30),
                "Behavior": current_behavior
            })

        # Record annotation
        behavior_annotations.append([
            "Individual1",
            current_behavior,
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            segment_end.strftime("%Y-%m-%d %H:%M:%S"),
            "Simulated"
        ])

        # Transition to next behavior
        current_index = behavior_to_index[current_behavior]
        current_behavior = np.random.choice(behaviors, p=transition_matrix[current_index])

        current_time = segment_end

    acc_df = pd.DataFrame(acc_data)
    annotations_df = pd.DataFrame(behavior_annotations, columns=["id", "Behavior", "Timestamp_start", "Timestamp_end", "Source"])
    return acc_df, annotations_df

def plot_simulated_day(acc_df, plot_path=None):
    acc_df['Timestamp'] = pd.to_datetime(acc_df['Timestamp'])

    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot X, Y, Z signals and store their handles
    signal_handles = []
    signal_handles.append(ax.plot(acc_df['Timestamp'], acc_df['Acc X [g]'], label='X', color='black', linewidth=.5, alpha=0.5)[0])
    signal_handles.append(ax.plot(acc_df['Timestamp'], acc_df['Acc Y [g]'], label='Y', color='blue', linewidth=.5, alpha=0.5)[0])
    signal_handles.append(ax.plot(acc_df['Timestamp'], acc_df['Acc Z [g]'], label='Z', color='maroon', linewidth=.5, alpha=0.5)[0])

    # Behavior segmenting
    acc_df['behavior_shift'] = acc_df['Behavior'].shift()
    acc_df['change'] = acc_df['Behavior'] != acc_df['behavior_shift']
    change_indices = acc_df.index[acc_df['change']].tolist() + [acc_df.index[-1] + 1]

    # Unique color for each behavior
    colors = dict(zip(acc_df['Behavior'].unique(), sns.color_palette("Set2", acc_df['Behavior'].nunique())))

    behavior_labels_used = set()
    behavior_handles = []

    for start_idx, end_idx in zip(change_indices[:-1], change_indices[1:]):
        row = acc_df.iloc[start_idx]
        behavior = row['Behavior']
        color = colors[behavior]

        # Only label the first time each behavior appears
        label = behavior if behavior not in behavior_labels_used else None
        span = ax.axvspan(acc_df.loc[start_idx, 'Timestamp'], acc_df.loc[end_idx - 1, 'Timestamp'],
                          alpha=0.3, color=color, label=label)
        if label:
            behavior_handles.append(span)
            behavior_labels_used.add(behavior)

    # Format axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude [g]')
    # ax.set_title('24 Hours of Simulated Acceleration Signal and Behavior Annotations')

    # Create legends
    legend1 = ax.legend(handles=signal_handles, loc='upper left')
    ax.add_artist(legend1)  # Add first legend manually
    ax.legend(handles=behavior_handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.))
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path, dpi=300)
    plt.show()
    


def generate_dataset(data_constants, class_distribution, window_length, n_samples, wrong_behavior=False, wrong_behavior_prob=0.4, tau=0.1):
    behaviors = list(class_distribution.keys())
    probabilities = list(class_distribution.values())

    def simulate_behavior_signal(behavior, window_length):

        signal = []
        for i in ['X', 'Y', 'Z']:
            f, A, phi, sigma = data_constants.loc[
                            (data_constants["Behavior"] == behavior) & 
                            (data_constants["Axis"] == i), ['f', 'A', 'phi', 'sigma']
                        ].values[0]

            signal.append(simulate_axis_signal(f, A, phi, sigma, window_length, tau=tau))

        return np.vstack(signal)
    
    X_list, y_list = [], []
    for _ in range(n_samples):

        behavior = np.random.choice(behaviors, p=probabilities)
        y_list.append(behavior)

        if wrong_behavior & (np.random.rand() < wrong_behavior_prob):
            behavior = config.WRONG_BEHAVIORS[behavior]

        signal = simulate_behavior_signal(behavior=behavior, window_length=window_length)
        X_list.append(signal)
    
    X = np.stack(X_list)  # shape: (N, 3, T)
    y = np.array(y_list)  # shape: (N,)
    return X, y

def compute_features(data):

    """
    Vectorized feature computation for shape (N, 3, T)
    Returns a pandas DataFrame with 38 named columns representing univariate summary statsitics of the time series
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        N, C, T = data.shape
        assert C == 3, "Expected shape (N, 3, T)"

        x, y, z = data[:, 0, :], data[:, 1, :], data[:, 2, :]
        q = np.sqrt(x**2 + y**2 + z**2)

        def autocorr1(ts):
            return np.array([
                np.corrcoef(ts[i, :-1], ts[i, 1:])[0, 1] if ts.shape[1] > 1 else np.nan
                for i in range(ts.shape[0])
            ])

        def trends(ts):
            t = np.arange(T)
            t_mean = np.mean(t)
            t_var = np.var(t)
            ts_mean = np.mean(ts, axis=1)
            cov = np.mean(ts * (t - t_mean), axis=1) - ts_mean * 0
            return cov / t_var

        def safe_skew(ts, axis):
            result = skew(ts, axis=axis)
            result[np.isnan(result)] = 0.0
            return result

        def safe_kurtosis(ts, axis):
            result = kurtosis(ts, axis=axis)
            result[np.isnan(result)] = 0.0
            return result


        def summary_stats(ts, name):
            return {
                f'{name}_mean': np.mean(ts, axis=1),
                f'{name}_std': np.std(ts, axis=1),
                f'{name}_skew': safe_skew(ts, axis=1),
                f'{name}_kurtosis': safe_kurtosis(ts, axis=1),
                f'{name}_max': np.max(ts, axis=1),
                f'{name}_min': np.min(ts, axis=1),
                f'{name}_autocorr': np.nan_to_num(autocorr1(ts), nan=0.0),
                # f'{name}_trend': trends(ts),
            }

        stats = {}
        for name, ts in zip(['x', 'y', 'z', 'q'], [x, y, z, q]):
            stats.update(summary_stats(ts, name))

        stats['corr_xy'] = np.nan_to_num(np.array([np.corrcoef(x[i], y[i])[0, 1] for i in range(N)]), nan=0.0)
        stats['corr_xz'] = np.nan_to_num(np.array([np.corrcoef(x[i], z[i])[0, 1] for i in range(N)]), nan=0.0)
        stats['corr_yz'] = np.nan_to_num(np.array([np.corrcoef(y[i], z[i])[0, 1] for i in range(N)]), nan=0.0)

        stats['odba'] = np.mean(
            np.abs(x - np.mean(x, axis=1, keepdims=True)) +
            np.abs(y - np.mean(y, axis=1, keepdims=True)) +
            np.abs(z - np.mean(z, axis=1, keepdims=True)),
            axis=1
        )

        q_safe = np.clip(q, 1e-8, None)
        theta = np.arccos(np.clip(z / q_safe, -1, 1))
        phi = np.arctan2(y, x)

        stats['inclination_circvar'] = np.array([circvar(theta[i], high=np.pi, low=0) for i in range(N)])
        stats['azimuth_circvar'] = np.array([circvar(phi[i], high=np.pi, low=-np.pi) for i in range(N)])

        # Combine into DataFrame
        df = pd.DataFrame(stats)
        return df

def energy_distance(X, Y):
    d_xy = cdist(X, Y).mean()
    d_xx = cdist(X, X).mean()
    d_yy = cdist(Y, Y).mean()
    return 2 * d_xy - d_xx - d_yy
