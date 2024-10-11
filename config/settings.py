import pytz

# These global variables should be modified based on the path and structure of your data.

VECTRONICS_BEHAVIOR_EVAL_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/behavior_evaluations"

AWD_VECTRONICS_PATHS = {'jessie': "/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44934_Samurai_Jessie",
 'green': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44915_Samurai_Green",
 'palus': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44910_Aqua_Palus",
 'ash': "/mnt/ssd/medhaaga/wildlife/Vectronics/2021_44904_Ninja_Ash",
 'fossey': '/mnt/ssd/medhaaga/wildlife/Vectronics/2022_44907_Aqua_Fossey'}

VECTRONICS_METADATA_PATH = "/mnt/ssd/medhaaga/wildlife/Vectronics/metadata.csv"

DATE_FORMAT = "%Y%m%d_%H%M%S"
TIMEZONE = pytz.utc
SAMPLING_RATE = 16

id_mapping = {'2021_ninja_ash': 'ash', '2021_aqua_palus': 'palus', '2021_samurai_green': 'green', 
            '2022_aqua_fossey': 'fossey', '2022_ninja_birch': 'birch', '2022_roman_bishop': 'bishop',
            '2022_samurai_jessie': 'jessie', '2022_royal_rossignol': 'rossignol', 'Jessie': 'jessie'}

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

BEHAVIORS = ['Feeding', 'Resting', 'Moving', 'Running', 'Vigilant']