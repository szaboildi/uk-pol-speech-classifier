# Download NLTK modules
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("wordnet")

# Set up folderstructure
import os
from polclassifier.params import *
if not os.path.isdir(os.path.join(LOCAL_PATH, "raw_data")):
    os.makedirs(os.path.join(LOCAL_PATH, "raw_data"))
    print("Created folder for raw data")

if not os.path.isdir(os.path.join(LOCAL_PATH, "processed_data")):
    os.makedirs(os.path.join(LOCAL_PATH, "processed_data"))
    print("Created folder for processed data")

if not os.path.isdir(os.path.join(LOCAL_REGISTRY_PATH, "vectorizers")):
    os.makedirs(os.path.join(LOCAL_REGISTRY_PATH, "vectorizers"))
    print("Created folder for vectorizers")

if not os.path.isdir(os.path.join(LOCAL_REGISTRY_PATH, "models")):
    os.makedirs(os.path.join(LOCAL_REGISTRY_PATH, "models"))
    print("Created folder for models")

if not os.path.isdir(os.path.join(LOCAL_REGISTRY_PATH, "text_plot")):
    os.makedirs(os.path.join(LOCAL_REGISTRY_PATH, "text_plot"))
    print("Created folder for Shapley plots")
