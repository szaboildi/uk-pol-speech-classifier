from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from polclassifier.interface.main import preprocess
from polclassifier.ml_logic.preprocessing import *
from polclassifier.params import *


def plot_confusion_matrix(model, vect_method="tfidf"):
    # Load data
    X, y = preprocess()
    
    if vect_method=="embed":
        if len(X.shape) > 1:
            X = X["text"]
            
    if len(y.shape) > 1:
        y = y["party"]
    labels = list(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    print("prediction made")

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# Example usage:
# Replace model, X_test, y_test, and labels with your trained model, test data, and labels respectively
# model = your trained scikit-learn model
# X_test = your test data
# y_test = true labels of your test data
# labels = list of label names

# plot_confusion_matrix(model, X_test, y_test, labels)

def load_speeches(min_word_count=400, sample_size=1000, parties_to_exclude=[], speeches_per_party = 20):
    # Load data from feather file
    raw_data_path = "~/code/uk-pol-speech-classifier/polclassifier/Corp_HouseOfCommons_V2.feather"
    try:
        data = pd.read_feather(raw_data_path)
        print("Data loaded successfully")
    except Exception as e:
        print("Error loading data:", e)
        return None

    # Filter and clean data
    data = clean_data(df=data, min_word_count=min_word_count, sample_size=sample_size, parties_to_exclude=parties_to_exclude)
    print(data)


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
    
    df = pd.concat(smaller_data_test, ignore_index=True)
    path="~/code/uk-pol-speech-classifier/polclassifier/smaller_data_test.csv"
    df.to_csv(path, index=False)

    return 

if __name__=="__main__":
    load_speeches()