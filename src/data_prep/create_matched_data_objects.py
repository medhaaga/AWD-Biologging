import sys
sys.path.append('../')
from tqdm import tqdm
from pytz import timezone
import pandas as pd
import numpy as np
from config.settings import SAMPLING_RATE

""" acc_data

Save a summary of the matched accelerations with behavioral annotations.

Columns 
------
id: string
    Name of dog

behavior: string
   Behavioral annotation

behavior_start: pd.datetime like object
    Timestamp for start of behavior

behavior_end: pd.datetime like object
    Timestamp for end of behavior

duration: float 
    Duration of behavior in seconds

date_am_pm_id: str
    date and am/pm of the behavior

acc_x: list[float]
    list of acceleration along x axis

acc_y: list[float]
    list of acceleration along x axis_y

acc_z: list[float]
    list of acceleration along x axis

"""

""" acc_summary

Save a summary of the matched accelerations with behavioral annotations.

Columns 
------
id: string
    Name of dog

date_am_pm_id: pd.datetime like object
    The object gives a unique ID to a half day.

annotations: float
    Duration of annotations available for the dog in the particlar half day.

acc: float
    Duration of acceleration data matching with annotations in the particlar half day

number of matched acc: int
    Number of behavioral annotations that were matched with acceleration data stream.


"""


def create_matched_data(filtered_metadata, annotations, verbose=True):
    """Match the files in metadata with available annotations

    Arguments 
    ---------------
    filtered_metadata: pandas dataframe 
    annotations: pandas dataframe

    Returns
    ----------------
    acc_summary: pandas dataframe = summary of the matched acceleration files
    acc_data: pandas dataframe = final matched windows of acc data 

    """

    # create dataframes for saving matched acceleration and behavior data

    cols = ['dog ID', 'behavior', 'behavior_start', 'behavior_end', 'duration', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm',  'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'acc_x', 'acc_y', 'acc_z', 'Source']
    acc_data = pd.DataFrame(columns=cols, index=[])
    acc_data_metadata = pd.DataFrame(columns=filtered_metadata.columns, index=[])
    acc_summary = pd.DataFrame(columns=['id', 'date_am_pm_id', 'annotations', 'acc', 'number of matched acc'], index=[])


    # loop over all unique dogs in filtered_metadata
    for (i, dog) in enumerate(filtered_metadata['dog ID'].unique()):

        ## find annotations for this dog
        annotations_orig = annotations[annotations['id'] == dog]
        dog_annotations = annotations_orig.copy()

        # Format and add helper columns to the annotations dataframe
        dog_annotations['Timestamp_start'] = pd.to_datetime(annotations_orig['Timestamp_start'], format='%Y/%m/%d %H:%M:%S')
        dog_annotations['Timestamp_end'] = pd.to_datetime(annotations_orig['Timestamp_end'], format='%Y/%m/%d %H:%M:%S')
        dog_annotations['date'] = dog_annotations['Timestamp_start'].dt.date
        dog_annotations['am/pm'] = dog_annotations['Timestamp_start'].dt.strftime('%P')
        dog_annotations['half day [yyyy-mm-dd_am/pm]'] = dog_annotations['date'].astype(str) + '_' + dog_annotations['am/pm']
        
        # create submetadata file for this dog
        dog_metadata = filtered_metadata[filtered_metadata['dog ID'] == dog]

        if verbose:
            print('Dog {} has {} halfdays in the filtered metadata.'.format(dog, len(dog_metadata)))

        for unique_period_loop in tqdm(dog_metadata['half day [yyyy-mm-dd_am/pm]'].unique(), desc=f'Processing unique half days for {dog}'):

            annotation_available = unique_period_loop in dog_annotations['half day [yyyy-mm-dd_am/pm]'].values

            if annotation_available:

                annotations_loop = dog_annotations[dog_annotations['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop]
                
                # if the acceleration file is available for this dog and half day, read it
                
                acc_file_path = dog_metadata.loc[dog_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'file path'].values[0]
                acc_loop = pd.read_csv(acc_file_path)
                acc_loop['Timestamp'] = pd.to_datetime(acc_loop['Timestamp'], format='mixed', utc=True)

                for _, row in annotations_loop.iterrows():
                        
                    behaviour_start_time = row['Timestamp_start'].to_pydatetime().replace(tzinfo=timezone('UTC'))
                    behaviour_end_time = row['Timestamp_end'].to_pydatetime().replace(tzinfo=timezone('UTC'))

                    if not pd.isnull(behaviour_end_time):
                            acc_summary.at[acc_summary.index[-1], 'annotations'] += (behaviour_end_time - behaviour_start_time).total_seconds() 
            
                    
                    if (not pd.isnull(behaviour_end_time)) & (behaviour_end_time > behaviour_start_time):          

                        # log the duration of audio avalilable for the behaviour

                        behaviour_acc = acc_loop[(acc_loop['Timestamp'] >= behaviour_start_time) & (acc_loop['Timestamp'] <= behaviour_end_time)].sort_values('Timestamp')
                        
                        # log the duration of acc avalilable for the behaviour
                        if len(behaviour_acc) > 0:

                            # duration of the acceleration data that is matched with the behavior
                            matched_duration = (behaviour_acc.iloc[len(behaviour_acc)-1]['Timestamp'] - behaviour_acc.iloc[0]['Timestamp']).total_seconds()
                            acc_summary.at[acc_summary.index[-1], 'acc'] += matched_duration
                            acc_summary.at[acc_summary.index[-1], 'number of matched acc'] += 1

                            acc_data.loc[len(acc_data)] = [dog, 
                                                            row['Behavior'], 
                                                            behaviour_acc.iloc[0]['Timestamp'], 
                                                            behaviour_acc.iloc[len(behaviour_acc)-1]['Timestamp'], 
                                                            matched_duration, 
                                                            dog_metadata.loc[dog_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'year'].values[0],
                                                            dog_metadata.loc[dog_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'UTC Date [yyyy-mm-dd]'].values[0],
                                                            dog_metadata.loc[dog_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'am/pm'].values[0],
                                                            unique_period_loop,
                                                            dog_metadata.loc[dog_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'avg temperature [C]'].values[0],
                                                            behaviour_acc['Acc X [g]'].to_list(),
                                                            behaviour_acc['Acc Y [g]'].to_list(),
                                                            behaviour_acc['Acc Z [g]'].to_list(),
                                                            row['Source']]

                            acc_data_metadata.loc[len(acc_data_metadata)] = dog_metadata.loc[dog_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop].values[0].tolist()
            
            else:
                acc_summary.loc[len(acc_summary)] = [dog, unique_period_loop, 0, 0.0, 0]

    return acc_summary, acc_data, acc_data_metadata



def repeat_or_truncate_list(lst, fixed_length, reuse=False, min_length=16):

    """
    Pad the list with repitition of data if it's shorter than fixed_length.
    Truncate the list if it's longer than fixed_length

    Arguments
    ---------
    lst: list 
    fixed_length: int
    reuse: indicator on whether to extract multiple windows from the list
    min_length: minimum length to repeat, discard otherwise
    
    """
    
    if reuse:
        bundle = []
        while len(lst) >= fixed_length:
                bundle.append(lst[:fixed_length])
                lst = lst[fixed_length:]
        if len(lst) >= min_length:
                lst = lst*(fixed_length//len(lst)) 
                bundle.append(lst + lst[:(fixed_length - len(lst))])
                
        return bundle

    else:
        if len(lst) < fixed_length:
                # Pad the list with 0 if it's shorter than fixed_length
                lst = lst*(fixed_length//len(lst)) 
                return [lst + lst[:(fixed_length - len(lst))]]

        else:
                # Truncate the list if it's longer than fixed_length
                return [lst[len(lst)-fixed_length:]]


def pad_or_truncate_list(lst, fixed_length):
    """
    Pad the list with 0 if it's shorter than fixed_length.
    Truncate the list if it's longer than fixed_length

    Arguments
    ---------
    lst: list 
    fixed_length: int
    
    """
    if len(lst) < fixed_length:
        # Pad the list with 0 if it's shorter than fixed_length
        return [0] * (fixed_length - len(lst)) + lst
    else:
        # Truncate the list if it's longer than fixed_length
        return lst[len(lst)-fixed_length:]

def create_padded_or_truncated_data(df, fixed_length, padding='repeat', reuse_behaviors=[], min_duration=2):
        
    """Load the dataset and make the acc sequence along x, y, z of fixed length.

    Arguments
    ---------
    df: Pandas DataFrame = dataframe with columns `acc_x`, `acc_y`, `acc_z`, and `behavior`.
    fixed_length: int = fix the length of accelerometer sequences
    padding: str = type of padding for accelerations smaller than fixed length
    reuse_behaviors: 

    Return 
    ------
    X: array-like objective (N, C, L)
        independent variables
        N is number of observations, C is number of input channels, L is the length of features 

    Y: array-like objective (N,)
        categorical labels
    """

    # Apply pad_or_truncate_list to all cells in columns x, y, z

    if len(df) == 0:
        raise ValueError('No data provided')


    df_new = pd.DataFrame(columns=['acc_x', 'acc_y', 'acc_z'])
    df_metadata = pd.DataFrame(columns=['dog ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'Source'])
    
    if padding == 'zeros':
        df_new['acc_x'] = df['acc_x'].apply(pad_or_truncate_list, args=(fixed_length,))
        df_new['acc_y'] = df['acc_y'].apply(pad_or_truncate_list, args=(fixed_length,))
        df_new['acc_z'] = df['acc_z'].apply(pad_or_truncate_list, args=(fixed_length,))
        
    elif padding == 'repeat':

        expanded_rows = []
        for _, row in df.iterrows():

            reuse = row['behavior'] in reuse_behaviors

            acc_x_windows = repeat_or_truncate_list(row['acc_x'], fixed_length, reuse=reuse, min_length=min_duration*SAMPLING_RATE)
            acc_y_windows = repeat_or_truncate_list(row['acc_y'], fixed_length, reuse=reuse, min_length=min_duration*SAMPLING_RATE)
            acc_z_windows = repeat_or_truncate_list(row['acc_z'], fixed_length, reuse=reuse, min_length=min_duration*SAMPLING_RATE)

            assert len(acc_x_windows) == len(acc_y_windows) == len(acc_z_windows) 

            for x, y, z in zip(acc_x_windows, acc_y_windows, acc_z_windows):
                expanded_rows.append({'acc_x': x, 'acc_y': y, 'acc_z': z, 'behavior': row['behavior']})
                df_metadata.loc[len(df_metadata)] = row[['dog ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'Source']].values

        df_new = pd.DataFrame(expanded_rows)
    else:
        raise ValueError


    arr_list = []
    for col in ['acc_x', 'acc_y', 'acc_z']:
        arr_list.append(np.array(df_new[col].to_list()))


    X = np.stack(arr_list, axis=2)
    X = np.transpose(X, (0, 2, 1))

    y = df_new['behavior'].values

    return X, y, df_metadata