import sys
import os
sys.path.append('.')
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def combined_annotations(video_path, audio_path, id_mapping):

    """Combine the annotations from gold and silver labels.
    
    Arguments 
    --------------------
    path: dictionary = a sictionary of paths to folders of different AWD
    id_mapping: dictionary =  dict for matching the id names in annotations to names used uniformly
    
    Returns 
    --------------------
    all_annotations: Pandas dataframe = dataframe of all annotations with time stamps

    """
    video_annotations = pd.read_csv(video_path) # load video annotations
    audio_annotations = pd.read_csv(audio_path) # load audio annotations

    video_annotations['id'] = video_annotations['id'].replace(id_mapping)
    audio_annotations['Individual'] = audio_annotations['Individual'].replace(id_mapping)

    audio_annotations = audio_annotations[audio_annotations['Confidence (H-M-L)'].isin(['H', 'H/M'])]
    
    audio_annotations = audio_annotations.assign(Source='Audio')
    video_annotations = video_annotations.assign(Source='Video')


    annotations_columns = ['id', 'Behavior', 'Timestamp_start', 'Timestamp_end', 'Source']
    rename_dict = {'Individual': 'id', 'Behaviour': 'Behavior', 'Timestamp_start_utc': 'Timestamp_start', 'Timestamp_end_utc': 'Timestamp_end'}
    audio_annotations = audio_annotations.rename(columns=rename_dict)

    botswana_timezone = 'Africa/Gaborone'
    video_annotations['Timestamp_start'] = pd.to_datetime(video_annotations['Timestamp_start'], format='%Y/%m/%d %H:%M:%S')
    video_annotations['Timestamp_end'] = pd.to_datetime(video_annotations['Timestamp_end'], format='%Y/%m/%d %H:%M:%S')

    # localize to botswana time and the change clock time to utc
    video_annotations['Timestamp_start'] = video_annotations['Timestamp_start'].apply(lambda x: x.tz_localize(botswana_timezone).tz_convert('UTC'))
    video_annotations['Timestamp_end'] = video_annotations['Timestamp_end'].apply(lambda x: x.tz_localize(botswana_timezone).tz_convert('UTC'))
    
    # now remove time zone information
    video_annotations['Timestamp_start'] = video_annotations['Timestamp_start'].dt.tz_localize(None)
    video_annotations['Timestamp_end'] = video_annotations['Timestamp_end'].dt.tz_localize(None)

    # make sure timestamp is in a given format
    video_annotations['Timestamp_start'] = video_annotations['Timestamp_start'].dt.strftime('%Y/%m/%d %H:%M:%S')
    video_annotations['Timestamp_end'] = video_annotations['Timestamp_end'].dt.strftime('%Y/%m/%d %H:%M:%S')


    all_annotations = pd.concat([video_annotations[annotations_columns], audio_annotations[annotations_columns]])

    return all_annotations

def filter_data(metadata, filter_profile):

        """Filters the index from metadata that satisfy metadata constraints.

        Arguments
        --------------
        metadata: pd.DataFrame
        filter_profile: dictionary-like object
        """

        filter_idx = np.arange(len(metadata))

        # filter desired individual ID
        if filter_profile['individual ID'] is not None:
                assert isinstance(filter_profile['individual ID'], list), "individual ID filter should be a list"
                filter_idx = [idx for idx in filter_idx if metadata.iloc[idx]['individual ID'] in filter_profile['individual ID']]

        # filter desired year
        if filter_profile['year'] is not None:
                assert isinstance(filter_profile['year'], list), "year filter should be a list"
                filter_idx = [idx for idx in filter_idx if metadata.iloc[idx]['year'] in filter_profile['year']]

        # filter desired dates
        if filter_profile['UTC Date [yyyy-mm-dd]'] is not None:
                assert isinstance(filter_profile['UTC Date [yyyy-mm-dd]'], list), "year filter should be a list"

                date_idx = []
                for date_range in filter_profile['UTC Date [yyyy-mm-dd]']:
                        assert date_range[0] is None or date_range[1] is None or date_range[0] < date_range[1], "Incorrect date range"
                        assert isinstance(date_range, tuple), "Each entry of the date filter should be a tuple of lower and upper range of desired dates. Provide -1 for entire tail."
                        range_idx = filter_idx
                        if date_range[0] is not None:
                                lower_limit = datetime.strptime(date_range[0], '%Y-%m-%d')
                                range_idx = [idx for idx in filter_idx if datetime.strptime(metadata.iloc[idx]['UTC Date [yyyy-mm-dd]'], "%Y-%m-%d") >= lower_limit]
                        
                        if date_range[1] is not None:
                                upper_limit = datetime.strptime(date_range[1], '%Y-%m-%d')
                                range_idx = [idx for idx in range_idx if datetime.strptime(metadata.iloc[idx]['UTC Date [yyyy-mm-dd]'], "%Y-%m-%d") <= upper_limit]
                        date_idx.extend(range_idx)
                filter_idx = date_idx

        # filter desired am/pd ID
        if filter_profile['am/pm'] is not None:
                assert isinstance(filter_profile['am/pm'], list), "am/pm filter should be a list"
                filter_idx = [idx for idx in filter_idx if metadata.iloc[idx]['am/pm'] in filter_profile['am/pm']]

        # filter desired half days
        if filter_profile['half day [yyyy-mm-dd_am/pm]'] is not None:
                assert isinstance(filter_profile['half day [yyyy-mm-dd_am/pm]'], list), "half day [yyyy-mm-dd_am/pm] filter should be a list"
                half_day_idx = []
                for date_range in filter_profile['half day [yyyy-mm-dd_am/pm]']:
                        assert date_range[0] is None or date_range[1] is None or date_range[0] < date_range[1], "Incorrect date range"

                        range_idx = filter_idx
                        if date_range[0] is not None:
                                range_idx = [idx for idx in filter_idx if metadata.iloc[idx]['half day [yyyy-mm-dd_am/pm]'] >= date_range[0]]
                        
                        if date_range[1] is not None:
                                range_idx = [idx for idx in range_idx if metadata.iloc[idx]['half day [yyyy-mm-dd_am/pm]'] <= date_range[1]]
                        half_day_idx.extend(range_idx)
                filter_idx = half_day_idx

        # filter desired average temperature ranges
        if filter_profile['avg temperature [C]'] is not None:
                assert isinstance(filter_profile['avg temperature [C]'], list), "avg temperature [C] filter should be a list"

                temp_idx = []
                for temp_range in filter_profile['avg temperature [C]']:

                        assert temp_range[0] is None or temp_range[1] is None or temp_range[0] < temp_range[1], "Incorrect temp range"
                        range_idx = filter_idx
                        if temp_range[0] is not None:
                                range_idx = [idx for idx in filter_idx if metadata.iloc[idx]['avg temperature [C]'] >= temp_range[0]]
                        
                        if temp_range[1] is not None:
                                range_idx = [idx for idx in range_idx if metadata.iloc[idx]['avg temperature [C]'] <= temp_range[1]]
                        temp_idx.extend(range_idx)
                filter_idx = temp_idx

        return filter_idx

def apply_band_pass_filter(data, cutoff_frequency=0.0, sampling_rate=16, btype='high', N=5, axis=2):

    """Apply a high, low, or band pass filter

    data: Numpy 3-D array 
    cutoff_frequency: Optional = scalar for high or low pass, array of size 2 for bandpass
    sampling_rate: int 
    btype: str = type of bandpass filter. Choices = ['high', 'low', 'bandpass']
    N: int = order of bandpass cutoff
    axis: int = axis along the temporal component of the time series
    
    """

    if cutoff_frequency == 0.0 and btype=='high':
        return data

    if cutoff_frequency == 0.5 * sampling_rate and btype=='low':
        return data

    nyquist = 0.5 * sampling_rate

    # High/Low-pass Butterworth filter
    if btype in ['high', 'low']:
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(N=N, Wn=normal_cutoff, btype=btype, analog=False)

    elif btype == 'bandpass':
        assert len(cutoff_frequency) == 2
        low = cutoff_frequency[0]/nyquist
        high = cutoff_frequency[1]/nyquist
        b, a = butter(N=N, Wn=[low, high], btype='band', analog=False)
    
    # Apply the filter to each time series in the data
    filtered_data = filtfilt(b, a, data, axis=axis)
    
    return filtered_data
