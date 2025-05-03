# import libraries
import sys
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

def simulate_axis_signal(f, A, phi, sigma, n_samples):

    f = [freq + np.random.uniform(-0.1, 0.1) for freq in list(f)]
    A = [amp + np.random.uniform(-0.1, 0.1) for amp in list(A)]
    phi = [ph + np.random.uniform(-0.1, 0.1) for ph in list(phi)]
    sigma = sigma + np.random.uniform(-0.1, 0.1, size=n_samples)

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


def simulate_markov_acc_day(data_constants, transition_matrix, avg_durations):
    
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
            
            return simulate_axis_signal(f, A, phi, sigma, n_samples)
        
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

def plot_simulated_day(acc_df):
    # Ensure timestamp is datetime
    acc_df['Timestamp'] = pd.to_datetime(acc_df['Timestamp'])

    # Plot signal
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(acc_df['Timestamp'], acc_df['Acc X [g]'], label='X Signal', color='black', linewidth=.5, alpha=0.5)
    ax.plot(acc_df['Timestamp'], acc_df['Acc Y [g]'], label='Y Signal', color='blue', linewidth=.5, alpha=0.5)
    ax.plot(acc_df['Timestamp'], acc_df['Acc Z [g]'], label='Z Signal', color='maroon', linewidth=.5, alpha=0.5)


    # Get start and end indices for each behavior block
    acc_df['behavior_shift'] = acc_df['Behavior'].shift()
    acc_df['change'] = acc_df['Behavior'] != acc_df['behavior_shift']
    change_indices = acc_df.index[acc_df['change']].tolist() + [acc_df.index[-1] + 1]

    # Pick distinct colors for behaviors
    import seaborn as sns
    colors = dict(zip(acc_df['Behavior'].unique(), sns.color_palette("Set2", acc_df['Behavior'].nunique())))

    # Mark regions using axvspan
    for start_idx, end_idx in zip(change_indices[:-1], change_indices[1:]):
        row = acc_df.iloc[start_idx]
        behavior = row['Behavior']
        color = colors[behavior]
        ax.axvspan(acc_df.loc[start_idx, 'Timestamp'], acc_df.loc[end_idx - 1, 'Timestamp'],
                alpha=0.3, color=color, label=behavior)

    # Format x-axis and legend
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.set_title('Signal over Time with Behavior Annotations')

    # To avoid duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, -1.2))

    plt.tight_layout()
    plt.show()