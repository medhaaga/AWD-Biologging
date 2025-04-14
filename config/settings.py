import pytz
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
