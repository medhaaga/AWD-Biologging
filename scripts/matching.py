import sys
sys.path.append('.')

# Data handling
import pandas as pd

# import scripts

from config.settings import (VECTRONICS_METADATA_PATH, 
                    VECTRONICS_VIDEO_ANNOTATIONS_PATH,
                    VECTRONICS_AUDIO_ANNOTATIONS_PATH, 
                    id_mapping
                    )
from src.data_prep.data_prep_utils import combined_annotations
from src.data_prep.create_matched_data_objects import create_matched_data
from src.utils.io import (get_matched_data_path,
                                get_matched_metadata_path,
                                get_matched_summary_path)

def match_behaviors(metadata_path, video_annotations_path, audio_annotations_path):
    """
    Matches accelerometer metadata with video and audio behavioral annotations, then saves the processed data as CSV files.

    Args:
        metadata_path (str): Path to the accelerometer metadata CSV file.
        video_annotations_path (str): Path to the video annotations file.
        audio_annotations_path (str): Path to the audio annotations file.

    Output:
        Saves three CSV files containing matched summary data, full accelerometer data, and metadata.
    """

    metadata = pd.read_csv(metadata_path)

    all_annotations = combined_annotations(video_annotations_path, audio_annotations_path, id_mapping)
    acc_summary, acc_data, acc_data_metadata = create_matched_data(metadata, all_annotations)

    acc_summary.to_csv(get_matched_summary_path(), index=False)
    acc_data.to_csv(get_matched_data_path(), index=False)
    acc_data_metadata.to_csv(get_matched_metadata_path(), index=False)


if __name__ == '__main__':

    match_behaviors(VECTRONICS_METADATA_PATH,
                    VECTRONICS_VIDEO_ANNOTATIONS_PATH,
                    VECTRONICS_AUDIO_ANNOTATIONS_PATH)