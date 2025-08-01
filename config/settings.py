import pytz
import pandas as pd

# GLOBAL VARIABLES 

# These global variables should be modified based on the path and structure of your data.

VECTRONICS_BEHAVIOR_EVAL_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/behavior_evaluations"

# dictionary with individual ID as key and path to the directory which stores that individual's acceleration CSV files as value.
AWD_VECTRONICS_PATHS = {'jessie': "/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44934_Samurai_Jessie",
 'green': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44915_Samurai_Green",
 'palus': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44910_Aqua_Palus",
 'ash': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44904_Ninja_Ash",
 'fossey': '/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44907_Aqua_Fossey'}

# path to metadata and behavior annotations file. 
# If two different sources of annotations exist, like audio and video, provide these paths here.
VECTRONICS_METADATA_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/metadata.csv"
VECTRONICS_VIDEO_ANNOTATIONS_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/annotations_combined.csv"
VECTRONICS_AUDIO_ANNOTATIONS_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/silver_labels_annotations.csv"
VECTRONICS_ANNOTATIONS_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/all_annotations.csv"

# don't change these
DATE_FORMAT = "%Y%m%d_%H%M%S"
TIMEZONE = pytz.utc
SAMPLING_RATE = 16

# no need to change these. We use it to map a separate encoding of individual ID to the globally used individual IDs
id_mapping = {'2021_ninja_ash': 'ash', '2021_aqua_palus': 'palus', '2021_samurai_green': 'green', 
            '2022_aqua_fossey': 'fossey', '2022_ninja_birch': 'birch', '2022_roman_bishop': 'bishop',
            '2022_samurai_jessie': 'jessie', '2022_royal_rossignol': 'rossignol', 'Jessie': 'jessie'}

# map fine behavior classifications in annottaions file to coarser behavior classes
COLLAPSE_BEHAVIORS_MAPPING = {'Lying (head up)': 'Vigilant', 
                                    'Lying (head down)': 'Resting',
                                    'Walking': 'Moving',
                                    'Trotting': 'Moving',
                                    'Running': 'Running',
                                    'Standing': 'Vigilant',
                                    'Sitting':  'Vigilant',
                                    'Marking (scent)': 'Marking',
                                    'Interaction': 'Other',
                                    'Rolling': 'Marking',
                                    'Scratching': 'Other',
                                    'Drinking': 'Other',
                                    'Dig': 'Other',
                                    'Capture?': 'Other',
                                    'Eating': 'Feeding',
                                    }

# behaviors of interest for classification 
BEHAVIORS = ['Feeding', 'Resting', 'Moving', 'Running', 'Vigilant']
SIM_BEHAVIORS = ['Feeding', 'Moving', 'Resting', 'Vigilant']

# test paths

# WILDLIFE_DIR = "/Users/medha/Documents/GitHub/AWD-Biologging"
WILDLIFE_DIR = "/home/medhaaga/AWD-Biologging"
TEST_ROOT_DIR = WILDLIFE_DIR+"/test"
TEST_PATHS = {"individual1":  TEST_ROOT_DIR+"/individual1",
                "individual2":  TEST_ROOT_DIR+"/individual2",
                "individual3":  TEST_ROOT_DIR+"/individual3"}
TEST_METADATA_PATH = TEST_ROOT_DIR+"/test_metadata.csv"
TEST_VIDEO_ANNOTATIONS_PATH = TEST_ROOT_DIR+"/test_video_labels_annotations.csv"
TEST_AUDIO_ANNOTATIONS_PATH = TEST_ROOT_DIR+"/test_audio_labels_annotations.csv"
TEST_ANNOTATIONS_PATH = TEST_ROOT_DIR+"/test_all_annotations.csv"

sim_constants = [
    
    # Feeding — bring down the amplitude and overlap with Moving
    ("Feeding", "X", (1.0, 3.0, 5.0), (1.0, 2.0, 3.0), (0.2, 0.7, 1.2), 1.0),
    ("Feeding", "Y", (2.0, 3.0, 4.0), (2.0, 3.0, 4.0), (0.3, 0.8, 1.3), 1.0),
    ("Feeding", "Z", (3.5, 4.5, 5.5), (3.5, 4.5, 5.5), (0.4, 0.9, 1.4), 1.0),

    # Moving — slightly reduce amplitude, overlap more with Feeding
    ("Moving", "X", (1.4, 3.4, 5.4), (1.4, 3.4, 5.4), (0.6, 1.2, 1.8), 1.0),
    ("Moving", "Y", (2.4, 3.4, 4.4), (2.4, 3.4, 4.4), (0.5, 1.0, 1.5), 1.0),
    ("Moving", "Z", (3.1, 4.1, 5.1), (3.1, 4.1, 5.1), (0.7, 1.2, 1.8), 1.0),

    # Resting — make it slightly jittery to overlap with Vigilant
    ("Resting", "X", (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.0, 0.1, 0.2), 0.5),
    ("Resting", "Y", (1.0, 1.5, 2.0), (1.0, 1.5, 2.0), (0.0, 0.1, 0.2), 0.5),
    ("Resting", "Z", (1.5, 2.0, 2.5), (1.5, 2.0, 2.5), (0.0, 0.1, 0.2), 0.5),

    # Vigilant — lower the amplitude and raise noise to blur with Resting
    ("Vigilant", "X", (0.8, 1.3, 1.8), (0.8, 1.3, 1.8), (0.0, 0.3, 0.6), 0.5),
    ("Vigilant", "Y", (0.7, 1.2, 1.7), (0.7, 1.2, 1.7), (0.0, 0.4, 0.8), 0.5),
    ("Vigilant", "Z", (1.8, 2.3, 2.8), (1.8, 2.3, 2.8), (0.0, 0.5, 1.0), 0.5),
]

SIMULATION_CONSTANTS = pd.DataFrame(sim_constants, columns=[
    "Behavior", "Axis", "f", "A", "phi", "sigma"
])

WRONG_BEHAVIORS = {'Feeding': 'Feeding',
                    'Moving': 'Feeding', 
                    'Resting': 'Vigilant', 
                    'Vigilant': 'Resting'}

quickstart_constants = [
    
    # Feeding — bring down the amplitude and overlap with Moving
    ("Feeding", "X", (1.0, 3.0, 5.0), (1.0, 2.0, 3.0), (0.2, 0.7, 1.2), 1.0),
    ("Feeding", "Y", (2.0, 3.0, 4.0), (2.0, 3.0, 4.0), (0.3, 0.8, 1.3), 1.0),
    ("Feeding", "Z", (3.5, 4.5, 5.5), (3.5, 4.5, 5.5), (0.4, 0.9, 1.4), 1.0),

    # Moving — slightly reduce amplitude, overlap more with Feeding
    ("Moving", "X", (1.4, 3.4, 5.4), (1.4, 3.4, 5.4), (0.6, 1.2, 1.8), 1.0),
    ("Moving", "Y", (2.4, 3.4, 4.4), (2.4, 3.4, 4.4), (0.5, 1.0, 1.5), 1.0),
    ("Moving", "Z", (3.1, 4.1, 5.1), (3.1, 4.1, 5.1), (0.7, 1.2, 1.8), 1.0),

    # Resting — make it slightly jittery to overlap with Vigilant
    ("Resting", "X", (0.5, 1.0, 1.5), (0.5, 1.0, 1.5), (0.0, 0.1, 0.2), 0.5),
    ("Resting", "Y", (1.0, 1.5, 2.0), (1.0, 1.5, 2.0), (0.0, 0.1, 0.2), 0.5),
    ("Resting", "Z", (1.5, 2.0, 2.5), (1.5, 2.0, 2.5), (0.0, 0.1, 0.2), 0.5),

    # Running — make it slightly jittery to overlap with Vigilant
    ("Running", "X", (1.5, 2.0, 3.5), (1.5, 2.0, 2.5), (0.0, 0.1, 0.2), 0.5),
    ("Running", "Y", (2.0, 2.5, 2.0), (2.0, 2.5, 3.0), (0.0, 0.1, 0.2), 0.5),
    ("Running", "Z", (2.5, 3.0, 3.5), (2.5, 3.0, 3.5), (0.0, 0.1, 0.2), 0.5),

    # Vigilant — lower the amplitude and raise noise to blur with Resting
    ("Vigilant", "X", (0.8, 1.3, 1.8), (0.8, 1.3, 1.8), (0.0, 0.3, 0.6), 0.5),
    ("Vigilant", "Y", (0.7, 1.2, 1.7), (0.7, 1.2, 1.7), (0.0, 0.4, 0.8), 0.5),
    ("Vigilant", "Z", (1.8, 2.3, 2.8), (1.8, 2.3, 2.8), (0.0, 0.5, 1.0), 0.5),
]

QUICKSTART_CONSTANTS = pd.DataFrame(quickstart_constants, columns=[
    "Behavior", "Axis", "f", "A", "phi", "sigma"
])