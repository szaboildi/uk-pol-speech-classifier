import os
from scipy.stats import uniform, randint

##################  VARIABLES  ##################
# Github
PROJECT_LEAD = "szaboildi"
PROJECT_NAME = "uk-pol-speech-classifier"

########### Constants ###############
LOCAL_PATH = os.path.join(
    os.path.expanduser('~'), "code", PROJECT_LEAD, PROJECT_NAME)
LOCAL_REGISTRY_PATH = os.path.join(LOCAL_PATH, "training_outputs")

# Preprocessing variables
REPROCESS_BY_DEFAULT = False # Should raw data be reprocessed even if it's already cached

MIN_WORD_COUNT = 400
SAMPLE_SIZE = 1000
PARTIES_TO_EXCLUDE = [] # list of strings with party names to exclude
MAX_WORD_COUNT = 600
EXTRACT_FROM = "middle" # Possible values: "start", "middle", "end"
VECT_METHOD = "tfidf" # Possible values: "tfidf", "for_embed"

# Tfidf vectorizer params
MIN_DF = 5
MAX_DF = 0.85
MAX_FEATURES = 10000

# Randomized search parameters for model SVM
PENALTY_C = uniform(0.1, 10)
KERNEL = ['linear', 'poly', 'rbf', 'sigmoid']
GAMMA = ['scale', 'auto']
DEGREE = randint(1, 10)

# Parameters for default model SVM
KERNEL_DEFAULT = "linear"
GAMMA_DEFAULT = "scale"
C_DEFAULT = 4.2

# Not sure where to put this

MODEL_TARGET = "local"
GAR_IMAGE="polclassifier"


BUCKET_NAME = "polclassifier-jonahramchandani"
