import os
import sys
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../../')
import time
import pandas as pd
from config.settings import (AWD_VECTRONICS_PATHS, 
                             VECTRONICS_METADATA_PATH)

def process_chunk_vectronics(chunk, dog, file_dir):

    '''Save each chunk of acceleration data in respective half day segments

    Arguments
    --------------
    chunk: pd Dataframe
    dog: dog ID
    file_dir: path-like object = directory to save segments 

    '''

    chunk['Timestamp'] = pd.to_datetime(chunk['UTC Date[mm/dd/yyyy]'] + ' ' + chunk['UTC DateTime'] , format='%m/%d/%Y %H:%M:%S')
    chunk['Timestamp'] += pd.to_timedelta(chunk['Milliseconds'], unit='ms')
    chunk['date_am_pm_id'] = pd.to_datetime(chunk['UTC Date[mm/dd/yyyy]'], format='%m/%d/%Y').dt.date.astype(str) + '_' + chunk['Timestamp'].dt.strftime('%P')

    unique_half_days = chunk['date_am_pm_id'].unique()
    print(unique_half_days)

    for x in unique_half_days:
        
        df = chunk[chunk['date_am_pm_id'] == x]
        
        file_name = os.path.join(file_dir, '{}_{}.csv'.format(dog, x))

        if os.path.exists(file_name):
            existing_data = pd.read_csv(file_name)
            df = pd.concat([existing_data, df], ignore_index=True)
            print(f'Concatenated DOG: {dog} HALF DAY: {x}')
        else:
            print(f'Saved DOG: {dog} HALF DAY: {x}')
   
        df.to_csv(file_name, index=False)
        


def combine_acc_vectronics(dog, acc_filepaths):

    '''break the yearly csv files for each dog into chunks

    Arguments
    --------------
    dog: dog ID
    acc_filepaths: path-like object = directory to save segments 
    
    '''

    # loop over csv files for each year for each dog

    for path in acc_filepaths:

        print(f'Handling the csv <{path}>')

        file_dir = os.path.join(os.path.dirname(path), 'combined_acc')
        os.makedirs(file_dir, exist_ok=True)

        chunk_size = 10**6  # Adjust the chunk size based on your available memory

        # Use chunksize to read the file in smaller portions
        for chunk in pd.read_csv(path, skiprows=1, chunksize=chunk_size):
            year = os.path.basename(path).split('.')[0]
            chunk['UTC Date[mm/dd/yyyy]'] = chunk['UTC Date[mm/dd]'] + '/' + year
            process_chunk_vectronics(chunk, dog, file_dir)
            del chunk

        time.sleep(10)


def run_vectronics():
    
    dog_outputs = pd.DataFrame({'location': list(AWD_VECTRONICS_PATHS.values()),
                            'id': list(AWD_VECTRONICS_PATHS.keys())})

    dogs = dog_outputs['id'].values

    dog_acc_filepaths = [[os.path.join(dog_outputs.iloc[i]['location'], file) for file in os.listdir(dog_outputs.iloc[i]['location']) if file.endswith('csv')] for i in range(len(dog_outputs))]
        
    # Sample data for the loop
    data = zip(dogs, dog_acc_filepaths)
    
    for dog, acc_filepaths in data:
        print(dog, [os.path.basename(file) for file in acc_filepaths])
        combine_acc_vectronics(dog, acc_filepaths)

def create_metadata():

    ## Read in your combined annotations

    VECTRONICS_LOCATIONS = pd.DataFrame({'id': list(AWD_VECTRONICS_PATHS.keys()),
                                'location': list(AWD_VECTRONICS_PATHS.values()),
                                'combined_acc_location': [os.path.join(file, 'combined_acc') for file in list(AWD_VECTRONICS_PATHS.values())],
                                'Outputs_location': [os.path.join(file, 'Outputs') for file in list(AWD_VECTRONICS_PATHS.values())]}
                                )
    print(VECTRONICS_LOCATIONS['id'].values)

    metadata = pd.DataFrame(columns = ['file path', 'dog ID', 'year', 'UTC Date [yyyy-mm-dd]', 'am/pm', 'half day [yyyy-mm-dd_am/pm]', 'avg temperature [C]'])

    dogs = VECTRONICS_LOCATIONS['id'].values
    dog_acc_filepaths = [[os.path.join(VECTRONICS_LOCATIONS.iloc[i]['combined_acc_location'], file) for file in os.listdir(VECTRONICS_LOCATIONS.iloc[i]['combined_acc_location']) if file.endswith('csv')] for i in range(len(VECTRONICS_LOCATIONS))]
        
    # Sample data for the loop
    data = zip(dogs, dog_acc_filepaths)

    for dog, acc_filepaths in data:

        print('Dog {} has {} halfdays.'.format(dog, len(acc_filepaths)))

        for file_path in tqdm(acc_filepaths):

            basename = os.path.basename(file_path).split('.')[0]
            date = basename.split('_')[1]
            year = date.split('-')[0]
            am_pm = basename.split('_')[2]
            half_day = date + '_' + am_pm

            csv_file = pd.read_csv(file_path)
            avg_temp = csv_file['Temperature [Celsius]'].mean()

            metadata.loc[len(metadata)] = [file_path, dog, year, date, am_pm, half_day, avg_temp]

    metadata.to_csv(VECTRONICS_METADATA_PATH, index=False)

if __name__ == '__main__':

    run_vectronics()
    create_metadata()