import os
import sys
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../../')
import time
import pandas as pd
from config.settings import (AWD_VECTRONICS_PATHS, 
                             VECTRONICS_METADATA_PATH)

def process_chunk_vectronics(chunk, individual, file_dir, verbose=False):

    '''Save each chunk of acceleration data in respective half day segments

    Arguments
    --------------
    chunk: pd Dataframe
    individual: individual ID
    file_dir: path-like object = directory to save segments 

    '''

    # Expected columns - [UTC Date[mm/dd], UTC DateTime, Milliseconds, Acc X [g], Acc Y [g], Acc Z [g], Temperature [Celsius]]

    chunk['Timestamp'] = pd.to_datetime(chunk['UTC Date[mm/dd/yyyy]'] + ' ' + chunk['UTC DateTime'] , format='%m/%d/%Y %H:%M:%S')
    chunk['Timestamp'] += pd.to_timedelta(chunk['Milliseconds'], unit='ms')
    chunk['date_am_pm_id'] = pd.to_datetime(chunk['UTC Date[mm/dd/yyyy]'], format='%m/%d/%Y').dt.date.astype(str) + '_' + chunk['Timestamp'].dt.strftime('%P')

    unique_half_days = chunk['date_am_pm_id'].unique()
    print(f"{'Number of half days in chunk:':<30} {len(unique_half_days)}")

    for x in unique_half_days:
        
        df = chunk[chunk['date_am_pm_id'] == x]
        
        file_name = os.path.join(file_dir, '{}_{}.csv'.format(individual, x))

        if os.path.exists(file_name):
            existing_data = pd.read_csv(file_name)
            df = pd.concat([existing_data, df], ignore_index=True)
            if verbose:
                print(f'Concatenated individual: {individual} HALF DAY: {x}')
        else:
            if verbose:
                print(f'Saved individual: {individual} HALF DAY: {x}')
   
        df.to_csv(file_name, index=False)
        


def combine_acc_vectronics(individual, acc_filepaths, max_chunks=0):

    '''break the yearly csv files for each individual into chunks

    Arguments
    --------------
    individual: individual ID
    acc_filepaths: list of path-like object for the CSV files for the individual. 
                   The basename of files is the year data was collected in (example - [2022.csv, 2023.csv, 2024.csv])
    max_chunks: stop reading a csv after these many chunks
    
    '''

    # loop over csv files for each year for each individual

    for path in acc_filepaths:

        print(f"{'Handling the csv:':<30} {path}")

        file_dir = os.path.join(os.path.dirname(path), 'combined_acc')
        os.makedirs(file_dir, exist_ok=True)

        chunk_size = 10**6  # Adjust the chunk size based on your available memory
        num_chunks = 0

        # Use chunksize to read the file in smaller portions
        for chunk in pd.read_csv(path, skiprows=1, chunksize=chunk_size):
            print(chunk.columns)
            num_chunks += 1
            year = os.path.basename(path).split('.')[0]
            chunk['UTC Date[mm/dd/yyyy]'] = chunk['UTC Date[mm/dd]'] + '/' + year
            process_chunk_vectronics(chunk, individual, file_dir)
            del chunk

            if max_chunks > 0 and num_chunks == max_chunks:
                break

        time.sleep(10)


def run_vectronics(path_mappings, max_chunks=0):
    """
    create segments by reading accelerometer data in chunks. Saves the segments in a 
    directory titled "combined_acc" inside the data directory of each individual.
    
    Parameters:
    - path_mappings (dict): A dictionary where keys are individual names (str) and values are file paths (str) to their data directories.
    - max_chunks (int, optional): The maximum number of chunks to process per individual. Default is 0 (no limit).
    
    """
    
    individual_outputs = pd.DataFrame({'location': list(path_mappings.values()),
                            'id': list(path_mappings.keys())})

    individuals = individual_outputs['id'].values

    individual_acc_filepaths = [[os.path.join(individual_outputs.iloc[i]['location'], file) for file in os.listdir(individual_outputs.iloc[i]['location']) if file.endswith('csv')] for i in range(len(individual_outputs))]
        
    # Sample data for the loop
    data = zip(individuals, individual_acc_filepaths)
    
    for individual, acc_filepaths in data:
        print(f"{'Processing individual:':<30} {individual}")
        print(f"{'Files for this individual :':<30}", [os.path.basename(file) for file in acc_filepaths])
        combine_acc_vectronics(individual, acc_filepaths, max_chunks=max_chunks)
        print("")

def create_metadata(path_mappings, metadata_path):

    """
    Generates metadata from accelerometer data files for multiple individuals.

    Parameters:
    - path_mappings (dict): A dictionary where keys are individual names (str) and values are file paths (str) to their data directories.
    - metadata_path (str): The file path where the generated metadata CSV file will be saved.

    Metadata columns 
    -----------

    file path: string
        path-like object of where the half day segment is stored
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

    ## Read in your combined annotations

    VECTRONICS_LOCATIONS = pd.DataFrame({'id': list(path_mappings.keys()),
                                'location': list(path_mappings.values()),
                                'combined_acc_location': [os.path.join(file, 'combined_acc') for file in list(path_mappings.values())],
                                'Outputs_location': [os.path.join(file, 'Outputs') for file in list(path_mappings.values())]}
                                )

    metadata = pd.DataFrame(columns = ['file path', 'individual ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]'])

    individuals = VECTRONICS_LOCATIONS['id'].values
    individuals_acc_filepaths = [[os.path.join(VECTRONICS_LOCATIONS.iloc[i]['combined_acc_location'], file) for file in os.listdir(VECTRONICS_LOCATIONS.iloc[i]['combined_acc_location']) if file.endswith('csv')] for i in range(len(VECTRONICS_LOCATIONS))]
        
    # Sample data for the loop
    data = zip(individuals, individuals_acc_filepaths)

    for individual, acc_filepaths in data:

        print('individual {} has {} halfdays.'.format(individual, len(acc_filepaths)))

        for file_path in tqdm(acc_filepaths):

            basename = os.path.basename(file_path).split('.')[0]
            date = basename.split('_')[1]
            year = date.split('-')[0]
            am_pm = basename.split('_')[2]
            half_day = date + '_' + am_pm

            csv_file = pd.read_csv(file_path)
            avg_temp = csv_file['Temperature [Celsius]'].mean()

            metadata.loc[len(metadata)] = [file_path, individual, year, date, am_pm, half_day, avg_temp]

    metadata.to_csv(metadata_path, index=False)

if __name__ == '__main__':

    run_vectronics(AWD_VECTRONICS_PATHS)
    create_metadata(AWD_VECTRONICS_PATHS, VECTRONICS_METADATA_PATH)