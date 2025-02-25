from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='polclassifier',
      version="0.0.1",
      description="UK Political Speech Classifier",
      license="MIT",
      author="szaboildi",
      author_email="ies236@nyu.edu",
      #url="https://github.com/szaboildi/uk-pol-speech-classifier",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)

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

if not os.path.isdir(os.path.join(LOCAL_PATH, "training_outputs", "vectorizers")):
    os.makedirs(os.path.join(LOCAL_PATH, "training_outputs", "vectorizers"))
    print("Created folder for vectorizers")

if not os.path.isdir(os.path.join(LOCAL_PATH, "training_outputs", "models")):
    os.makedirs(os.path.join(LOCAL_PATH, "training_outputs", "models"))
    print("Created folder for models")

if not os.path.isdir(os.path.join(LOCAL_PATH, "training_outputs", "text_plot")):
    os.makedirs(os.path.join(LOCAL_PATH, "training_outputs", "text_plot"))
    print("Created folder for Shapley plots")
