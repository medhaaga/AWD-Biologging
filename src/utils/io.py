import os
import datetime

def get_project_root() -> str:
    """Returns the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_path(levels, main_dir):
    path = main_dir
    for item in levels:
        path = os.path.join(path, item + "/")
        if not os.path.exists(path):
            os.mkdir(path)
    return path

def get_matched_data_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'matched_acc_data.csv')
    return path

def get_matched_metadata_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'matched_acc_metadata.csv')
    return path

def get_matched_summary_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'matched_acc_summary.csv')
    return path

def get_metadata_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'metadata.csv')
    return path

def get_video_labels_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'video_labels.csv')
    return path

def get_audio_labels_path():
    data_path = os.path.join(get_project_root(), 'data')
    os.makedirs(data_path, exist_ok=True)
    path = os.path.join(data_path, 'audio_labels.csv')
    return path

def get_results_dir():
    current_path = get_project_root()
    path = os.path.join(current_path, 'results')
    os.makedirs(path, exist_ok=True)
    return path

def get_results_path(exp_name, n_CNNlayers, n_channels, kernel_size, theta, window_duration_percentile):
    results_dir = get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    levels = ['predictions', exp_name, 'conv_layers_'+str(n_CNNlayers), \
             'n_channels_'+str(n_channels), 'kernel_size_'+str(kernel_size), \
             'theta_'+str(theta), 'duration_'+str(window_duration_percentile)]
    return get_path(levels, results_dir)

def get_online_pred_path(halfday):
    results_dir = get_results_dir()
    levels = ['online_predictions', halfday]
    return get_path(levels, results_dir)

def get_figures_dir():
    current_path = get_project_root()
    path = os.path.join(current_path, 'figures')
    return path

# utility functions
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':
    print(get_results_path('no_split', 5, 32, 5, 0.0, 50))