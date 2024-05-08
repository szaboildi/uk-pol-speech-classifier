import os

##################  VARIABLES  ##################
# Github
PROJECT_LEAD = "szaboildi"
PROJECT_NAME = "uk-pol-speech-classifier"

# Preprocessing variables
REPROCESS_BY_DEFAULT = False # Should raw data be reprocessed even if it's already cached

MIN_WORD_COUNT = 400
SAMPLE_SIZE = 1000
PARTIES_TO_EXCLUDE = [] # list of strings with party names to exclude
MAX_WORD_COUNT = 600
EXTRACT_FROM = "middle" # Possible values: "start", "middle", "end"
VECT_METHOD = "tfidf" # Possible values: "tfidf"

# Tfidf vectorizer params
MIN_DF = 5
MAX_DF = 0.85
MAX_FEATURES = 10000

##################  CONSTANTS  #####################
LOCAL_PATH = os.path.join(
    os.path.expanduser('~'), "code", PROJECT_LEAD, PROJECT_NAME)
