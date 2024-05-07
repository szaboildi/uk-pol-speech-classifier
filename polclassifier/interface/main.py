import os
from colorama import Fore, Style
import pandas as pd


from polclassifier.ml_logic.preprocessing import *
from polclassifier.params import *


def preprocess():
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    X_path = os.path.join(
        LOCAL_PATH, "processed_data",
        f"features_{SAMPLE_SIZE}sample_{MIN_WORD_COUNT}min_{MAX_WORD_COUNT}cutoff.csv")
    y_path = os.path.join(
        LOCAL_PATH, "processed_data",
        f"target_{SAMPLE_SIZE}sample_{MIN_WORD_COUNT}min_{MAX_WORD_COUNT}cutoff.csv")

    # Check cache
    # if there, load from there
    if os.path.isfile(X_path) and os.path.isfile(y_path):
        print("✅ X and y loaded in from cache \n")
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
    # if not, do proper preprocessing
    else:
        # If the folder is also missing, make it first
        if not os.path.isdir(os.path.join(LOCAL_PATH, "processed_data")):
            os.mkdir(os.path.join(LOCAL_PATH, "processed_data"))

        raw_data_path = os.path.join(
            LOCAL_PATH, "raw_data", "Corp_HouseOfCommons_V2.feather")

        data = pd.read_feather(raw_data_path)

        print("✅ Raw dataset loaded in \n")

        X, y = preprocess_all(
            data,
            min_word_count=MIN_WORD_COUNT, sample_size=SAMPLE_SIZE,
            parties_to_exclude=PARTIES_TO_EXCLUDE,
            max_word_count=MAX_WORD_COUNT, extract_from=EXTRACT_FROM,
            min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES,
            vect_method=VECT_METHOD)

        print("✅ X and y preprocessed \n")

        save_processed_to_cache(
            X, y, local_path=LOCAL_PATH, sample_size=SAMPLE_SIZE,
            min_word_count=MIN_WORD_COUNT, max_word_count=MAX_WORD_COUNT)

        print("✅ X and y saved to cache \n")

    return X, y


if __name__ == '__main__':
    X, y = preprocess()
