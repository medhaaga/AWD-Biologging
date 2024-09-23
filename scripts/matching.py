import sys
sys.path.append('.')

# Data handling
import pandas as pd

# import scripts

from config.settings import (VECTRONICS_METADATA_PATH, 
                    AWD_VECTRONICS_PATHS, 
                    id_mapping
                    )
from src.data_prep.data_prep_utils import combined_annotations
from src.data_prep.create_matched_data_objects import create_matched_data
from src.utils.io import (get_matched_data_path,
                                get_matched_metadata_path,
                                get_matched_summary_path)

def match_behaviors():

    metadata = pd.read_csv(VECTRONICS_METADATA_PATH)
    all_annotations = combined_annotations(AWD_VECTRONICS_PATHS, id_mapping)
    acc_summary, acc_data, acc_data_metadata = create_matched_data(metadata, all_annotations)
    acc_summary.to_csv(get_matched_summary_path(), index=False)
    acc_data.to_csv(get_matched_data_path(), index=False)
    acc_data_metadata.to_csv(get_matched_metadata_path(), index=False)


if __name__ == '__main__':
    match_behaviors()