import sys
sys.path.append('../')
from tqdm import tqdm
from pytz import timezone
import pandas as pd
import numpy as np
from config.settings import SAMPLING_RATE



def create_matched_data(filtered_metadata, annotations, verbose=True):
    
    """Match the files in metadata with available annotations

    Arguments 
    ---------------
    filtered_metadata: pandas dataframe with columns ['file path', 'individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]']
    annotations: pandas dataframe with columns ['id', 'Behavior', 'Timestamp_start', 'Timestamp_end', 'Source']

    Returns
    ----------------
    acc_summary: pandas dataframe = summary of the matched acceleration files with columns 

        id: string
            individual ID
        date_am_pm_id: string
            In format yyyy-mm-dd_{am/pm} 
        annotations: string 
            behavior class
        acc: float
            matched acceleration duration
        number of matched acc: int 
            number of matched annotations in the half day

    acc_data: pandas dataframe = final matched windows of acc data  with columns

        individual ID: string
            individual ID
        behavior: string
            behvior annotation
        behavior_start: string 
            behavior start time in format %Y/%m/%d %H:%M:%S
        behavior_end: string 
            behavior end time in format %Y/%m/%d %H:%M:%S
        duration: float
            duration of the matched behavior
        year: int 
            year of behavior observation
        UTC Date [yyyy-mm-dd]: string 
            date of behavior observation
        am/pm: string 
            AM or PM time of behavior observation
        half day [yyyy-mm-dd_am/pm]: string 
            half day of behavior observation
        avg temperature [C]: float 
            average temperature on the half day of behavior observation
        acc_x: list-like object
            acceleration data along X axis
        acc_y: list-like object
            acceleration data along Y axis
        acc_z: list-like object 
            acceleration data along Z axis
        Source: string 
            source of behavior annotation (video, audio, etc)

    acc_data_metadata: pandas dataframe = metadata of the acceleration segments matched with annotations

        file_path: string
            file path where the half-day segment of the acceleration snippet is stored
        individual ID: string
            individual ID
        year: int 
            year of behavior observation
        UTC Date [yyyy-mm-dd]: string 
            date of behavior observation
        am/pm: string 
            AM or PM time of behavior observation
        half day [yyyy-mm-dd_am/pm]: string 
            half day of behavior observation
        avg temperature [C]: float 
            average temperature on the half day of behavior observation

    """
    # create dataframes for saving matched acceleration and behavior data

    cols = ['individual ID', 'behavior', 'behavior_start', 'behavior_end', 'duration', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm',  'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'acc_x', 'acc_y', 'acc_z', 'Source']
    acc_data = pd.DataFrame(columns=cols, index=[])
    acc_data_metadata = pd.DataFrame(columns=filtered_metadata.columns, index=[])
    acc_summary = pd.DataFrame(columns=['id', 'date_am_pm_id', 'annotations', 'acc', 'number of matched acc'], index=[])


    # loop over all unique individuals in filtered_metadata
    for (i, individual) in enumerate(filtered_metadata['individual ID'].unique()):

        ## find annotations for this individual
        annotations_orig = annotations[annotations['id'] == individual]
        individual_annotations = annotations_orig.copy()

        # Format and add helper columns to the annotations dataframe
        individual_annotations['Timestamp_start'] = pd.to_datetime(annotations_orig['Timestamp_start'], format='%Y/%m/%d %H:%M:%S')
        individual_annotations['Timestamp_end'] = pd.to_datetime(annotations_orig['Timestamp_end'], format='%Y/%m/%d %H:%M:%S')
        individual_annotations['date'] = individual_annotations['Timestamp_start'].dt.date
        individual_annotations['am/pm'] = pd.to_datetime(individual_annotations['Timestamp_start'], format="%Y/%m/%d %H:%M:%S").dt.strftime('%p') 
        individual_annotations['half day [yyyy-mm-dd_am/pm]'] = individual_annotations['date'].astype(str) + '_' + individual_annotations['am/pm']
        
        # create submetadata file for this individual
        individual_metadata = filtered_metadata[filtered_metadata['individual ID'] == individual]

        if verbose:
            print('individual {} has {} halfdays in the filtered metadata.'.format(individual, len(individual_metadata)))

        for unique_period_loop in tqdm(individual_metadata['half day [yyyy-mm-dd_am/pm]'].unique(), desc=f'Processing unique half days for {individual}'):

            annotation_available = unique_period_loop in individual_annotations['half day [yyyy-mm-dd_am/pm]'].values

            if annotation_available:

                annotations_loop = individual_annotations[individual_annotations['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop]
                
                # if the acceleration file is available for this individual and half day, read it
                
                acc_file_path = individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'file path'].values[0]
                acc_loop = pd.read_csv(acc_file_path)
                acc_loop['Timestamp'] = pd.to_datetime(acc_loop['Timestamp'], format='mixed', utc=True)

                for _, row in annotations_loop.iterrows():
                        
                    behaviour_start_time = row['Timestamp_start'].to_pydatetime().replace(tzinfo=timezone('UTC'))
                    behaviour_end_time = row['Timestamp_end'].to_pydatetime().replace(tzinfo=timezone('UTC'))

                    if not pd.isnull(behaviour_end_time):
                        acc_summary.loc[len(acc_summary)] = [individual, unique_period_loop, 0, 0.0, 0]
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

                            acc_data.loc[len(acc_data)] = [individual, 
                                                            row['Behavior'], 
                                                            behaviour_acc.iloc[0]['Timestamp'], 
                                                            behaviour_acc.iloc[len(behaviour_acc)-1]['Timestamp'], 
                                                            matched_duration, 
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'year'].values[0],
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'UTC Date [yyyy-mm-dd]'].values[0],
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'am/pm'].values[0],
                                                            unique_period_loop,
                                                            individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop, 'avg temperature [C]'].values[0],
                                                            behaviour_acc['Acc X [g]'].to_list(),
                                                            behaviour_acc['Acc Y [g]'].to_list(),
                                                            behaviour_acc['Acc Z [g]'].to_list(),
                                                            row['Source']]

                            acc_data_metadata.loc[len(acc_data_metadata)] = individual_metadata.loc[individual_metadata['half day [yyyy-mm-dd_am/pm]'] == unique_period_loop].values[0].tolist()
            
            else:
                acc_summary.loc[len(acc_summary)] = [individual, unique_period_loop, 0, 0.0, 0]

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
                # Repeat the list if it's shorter than fixed_length
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

def create_padded_or_truncated_data(df, fixed_length, padding='repeat', reuse_behaviors=[], min_duration=1.0):
        
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
    df_metadata = pd.DataFrame(columns=['individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'Source'])
    
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
                df_metadata.loc[len(df_metadata)] = row[['individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]', 'Source']].values

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