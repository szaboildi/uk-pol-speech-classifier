import os
import pandas as pd

from sklearn.model_selection import train_test_split
from polclassifier.ml_logic.preprocessing import *
from polclassifier.params import *

def load_speeches(min_word_count=400, sample_size=1000, parties_to_exclude=[], speeches_per_party = 20):
    import ipdb; ipdb.set_trace()
    # Load data from feather file
    raw_data_path = os.path.join(
            LOCAL_PATH, "raw_data", "Corp_HouseOfCommons_V2.feather")
    data = pd.read_feather(raw_data_path)

    # Filter and clean data
    data = clean_data(df=data, min_word_count=min_word_count, sample_size=sample_size, parties_to_exclude=parties_to_exclude)

    # Split the data into training and testing sets
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["party"])

    # Undersample data_test
    grouped_data = data_test.groupby('party')
    smaller_data_test = []
    for party, group in grouped_data:
        # Select randomly 20 speeches per party
        sampled_group = group.sample(n=speeches_per_party, random_state=42)
        # Add selected speeches to list
        smaller_data_test.append(sampled_group)

    return smaller_data_test

if __name__ == '__main__':
    #train_evaluate_model_knn()
    #train_evaluate_model_svm()
    load_speeches()
