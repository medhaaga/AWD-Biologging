import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Graphing Parameters
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 30
mpl.rcParams['ytick.labelsize'] = 30
mpl.rcParams["axes.labelsize"] = 40
mpl.rcParams['legend.fontsize'] = 40
mpl.rcParams['axes.titlesize'] = 40
mpl.rcParams['text.usetex'] = True
import config as config

def plot_raw_time_series(X, y, save_path=None):
    """
    Plot raw time series data for each unique class in y.

    Parameters:
    - X (np.ndarray): 3D array of time series data with shape (n_samples, n_channels, n_time_steps).
    - y (np.ndarray): Array of class labels corresponding to each sample in X.
    """

    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    fig, axs = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), sharex=True, sharey=True)
    if n_classes == 1:
        axs = [axs]

    custom_palette = sns.color_palette("husl", 3)
    handles = []
    labels = []

    for j, cls in enumerate(unique_classes):
        i = np.where(y == cls)[0][0]
        time = np.linspace(0, X.shape[2] / 16, X.shape[2])

        # Plot each channel with corresponding colors
        for k, label in enumerate(['X', 'Y', 'Z']):
            line, = axs[j].plot(time, X[i, k, :], label=label, color=custom_palette[k], linewidth=2)
            if j == 0:  # Collect handles and labels only from the first subplot
                handles.append(line)
                labels.append(label)

        # Set title for the subplot
        axs[j].set_title(f'{cls}')
        axs[j].set_xlabel('Time (s)')

    axs[0].set_ylabel('Amplitude (g)')
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.15))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_fourier_features(X, y, num_plots=1, fps=16, ylim=[None, None, None], standardize=False):

    T = X.shape[2]
    time = np.linspace(0, int(T/16), T)
    custom_palette = sns.color_palette("husl", 3)
    fft_result = np.abs(np.fft.fft(X, axis=2))

    if standardize:
        fft_result = (fft_result - np.mean(fft_result, axis=0, keepdims=True))/np.std(fft_result, axis=0, keepdims=True)

    unique_classes = np.unique(y)

    for j in unique_classes[:num_plots]:
        
        i = np.where(y == j)[0][0]
        

        # Plot the original time series and Fourier features for the first sample
        plt.figure(figsize=(4, 6))

        # Plot the original time series
        plt.subplot(4, 1, 1)
        plt.plot(time, X[i, 0, :], label='X', color=custom_palette[0], linewidth=1)
        plt.plot(time, X[i, 1, :], label='Y', color=custom_palette[1], linewidth=1)
        plt.plot(time, X[i, 2, :], label='Z', color=custom_palette[2], linewidth=1)
        plt.title(f'Original Time Series - {y[i]}')
        plt.legend(loc='upper left')

        # Plot the magnitudes of Fourier coefficients for the 'X' time series
        plt.subplot(4, 1, 2)
        plt.plot(np.fft.fftfreq(T, 1/fps), fft_result[i, 0, :],color=custom_palette[0], linewidth=1)
        plt.title('X Acc Fourier Features')
        if ylim[0] is not None:
            plt.ylim((0, ylim[0]))

        # Plot the magnitudes of Fourier coefficients for the 'Y' time series
        plt.subplot(4, 1, 3)
        plt.plot(np.fft.fftfreq(T, 1/fps), fft_result[i, 1, :], color=custom_palette[1], linewidth=1)
        plt.title('Y Acc Fourier Features')
        if ylim[1] is not None:
            plt.ylim((0, ylim[1]))

        # Plot the magnitudes of Fourier coefficients for the 'Z' time series
        plt.subplot(4, 1, 4)
        plt.plot(np.fft.fftfreq(T, 1/fps), fft_result[i, 2, :], color=custom_palette[2], linewidth=1)
        plt.title('Z Acc Fourier Features')
        if ylim[2] is not None:
            plt.ylim((0, ylim[2]))

        # Adjust layout
        plt.tight_layout()
        plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

def multi_label_predictions(dir, label_encoder, split='test', plot_confusion=True, return_accuracy=False, return_precision=False, return_recall=False, return_f1=False, plot_path=None, average='macro'):
    
    if split == 'test':
        y = np.load(os.path.join(dir, 'test_true_classes.npy'))
        predictions = np.load(os.path.join(dir, 'test_predictions.npy'))

    elif split == 'val':
        y = np.load(os.path.join(dir, 'val_true_classes.npy'))
        predictions = np.load(os.path.join(dir, 'val_predictions.npy'))
    else:
        raise ValueError

    # plot confusion matrices

    if plot_confusion:
        cm = confusion_matrix(y, predictions, normalize='true')
        class_names = label_encoder.inverse_transform(np.arange(len(np.unique(y))))

        plt.clf()

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, square=True, linewidths=0,
                    annot_kws={"size": 30},)

        ax.set_xlabel("Predicted Label", fontsize=30, labelpad=20)
        ax.set_ylabel("True Label", fontsize=30, labelpad=20)
        ax.set_xticklabels(class_names, fontsize=25, rotation=90)
        ax.set_yticklabels(class_names, fontsize=25, rotation=0)

        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path, format="png", bbox_inches="tight")
        plt.show()
    
    if return_accuracy:
        label_accuracies = accuracy_score(y, predictions) 
        return label_accuracies
    
    if return_precision:
        label_precisions = precision_score(y, predictions, average=average, zero_division=0) 
        return label_precisions
    
    if return_recall:
        label_recalls = recall_score(y, predictions, average=average, zero_division=0)
        return label_recalls
    
    if return_f1:
        label_f1s = f1_score(y, predictions, average=average, zero_division=0) 
        return label_f1s

def plot_signal(signal, sampling_rate):
    sns.set_style("whitegrid")
    plt.figure(figsize=(15,3))
    plt.plot(np.arange(0, len(signal))/sampling_rate, signal, color='cornflowerblue', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel("Amplitude (g)")
    plt.plot()

def plot_online_predictions(online_avg, window_length, hope_length, window_duration, label_encoder):

    x = window_duration*(np.arange(online_avg.shape[-1])*hope_length + window_length/2)/3600
    y = np.arange(online_avg.shape[0])

    X, Y = np.meshgrid(x, y)
    color_intensity = online_avg

    X_flat = X.flatten()  
    Y_flat = Y.flatten()  
    color_flat = color_intensity.flatten()  

    y_labels = label_encoder.inverse_transform(y)

    plt.figure(figsize=(15,3))

    plt.scatter(X_flat, Y_flat, c='grey', s=120, marker='s', alpha=0.2)  
    plt.scatter(X_flat, Y_flat, c=color_flat, cmap='Blues', s=140, marker='s', alpha=0.7)  
    plt.colorbar(label='Probability') 
    plt.xlabel("Time (h)")
    plt.ylabel("Behavior")
    plt.yticks(y, y_labels)
    plt.ylim(-1,6)
    plt.tight_layout()
    plt.show()

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_signal_and_online_predictions(time, signal, online_avg, window_length, hop_length, window_duration, label_encoder, sampling_rate=16, plot_dir=None, half_day_behaviors=None):
    """
    Plots the raw signal and the online predictions.

    Parameters:
    ---------------
    - time: time stamps of the signal (1D array).
    - signal: The raw signal data (1D array).
    - online_avg: The average online prediction probabilities (2D array).
    - window_length: Length of each window for smoothening.
    - hop_length: Overlap length between windows in data points.
    - window_duration: Duration of each window in seconds.
    - label_encoder: A label encoder for behavior labels.
    - sampling_rate: The sampling rate of the signal (Hz).
    - plot_dir: Directory where the plot will be saved. If None, the plot is not saved.
    """
    
    sns.set_style("whitegrid")  # Set seaborn style at the beginning

    # Calculate x-axis in hours
    x = window_duration * (np.arange(online_avg.shape[-1]) * hop_length + window_length / 2) / 3600
    
    # y-axis labels for each row in online_avg
    y = np.arange(online_avg.shape[0])

    # Create a mesh grid for plotting
    X, Y = np.meshgrid(x, y)
    color_intensity = online_avg

    # Flatten the mesh grid and color intensity for scatter plot
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    color_flat = color_intensity.flatten()

    # Get inverse transformed labels for y-axis
    y_labels = label_encoder.inverse_transform(y)

    # Create figure and GridSpec
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[30, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.6)

    # Create subplots
    ax_signal = fig.add_subplot(gs[0, 0])
    ax_online = fig.add_subplot(gs[1, 0])
    cbar_ax = fig.add_subplot(gs[1, 1])

    # Plot the signal
    ax_signal.plot(time, signal[0,0,:], label='X Signal', color='black', linewidth=.5, alpha=0.6)
    ax_signal.plot(time, signal[0,1,:], label='Y Signal', color='blue', linewidth=.5, alpha=0.5)
    ax_signal.plot(time, signal[0,2,:], label='Z Signal', color='maroon', linewidth=.5, alpha=0.4)
    ax_signal.set_xlabel('Time (h)')
    ax_signal.set_ylabel("Amplitude (g)")
    ax_signal.set_title("Raw Signal along X axis")
    ax_signal.xaxis.set_major_locator(MultipleLocator(1))  # 1 hour intervals
    ax_signal.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # Show as integer hours


    # Plot behaviors
    if half_day_behaviors is not None:
        colors = dict(zip(config.BEHAVIORS, sns.color_palette("Set2", len(config.BEHAVIORS))))
        for i in range(len(half_day_behaviors)):
            behavior = half_day_behaviors.iloc[i]['behavior']
            ax_signal.axvspan(half_day_behaviors.iloc[i]['behavior_start'], 
                              half_day_behaviors.iloc[i]['behavior_end'], 
                              color=colors[behavior], 
                              alpha=0.3)
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[behavior], markersize=10, alpha=0.3, label=behavior)
                  for behavior in label_encoder.classes_]
    ax_signal.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.26, 1.14), fontsize=25)


    # Plot the online predictions
    scatter = ax_online.scatter(X_flat, Y_flat, c=color_flat, cmap='Blues', s=140, marker='s', alpha=0.7)
    ax_online.set_xlabel("Time (h)")
    ax_online.set_yticks(y)
    ax_online.set_yticklabels(y_labels)
    ax_online.set_ylim(-1, len(y))
    ax_online.set_title(f"Online Predictions, $s = {window_length}$")

    # Add colorbar to the scatter plot, placed in the separate axis
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Softmax Score', fontsize=25)

    # Adjust layout to fit everything
    plt.tight_layout(rect=[0, 0, 2.9, 1])  # Reduce right space for colorbar

    # Save plot if plot_dir is specified
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)  # Create directory if it doesn't exist
        plt.savefig(os.path.join(plot_dir, f'window_length_{window_length}.png'), dpi=300)
    
    plt.show()

    return fig, (ax_signal, ax_online)

from matplotlib import gridspec
from matplotlib.lines import Line2D