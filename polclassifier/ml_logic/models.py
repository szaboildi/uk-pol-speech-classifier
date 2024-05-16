import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC

from tensorflow.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from colorama import Fore, Style
import joblib

from polclassifier.params import *


####### SVM #######
def randomized_search_model_svm(X, y):
    print("Grid searching SVM model\n")
    # Define the hyperparameter grid
    param_grid = {
    'C': PENALTY_C,  # Penalty parameter C (regularization parameter)
    'kernel': KERNEL,  # Kernel type
    'gamma': GAMMA,  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    'degree': DEGREE  # Degree of the polynomial kernel function ('poly')
    }

    # Create an SVM classifier
    svm_classifier = SVC()

    # Perform random search cross-validation
    random_search = RandomizedSearchCV(
        svm_classifier, param_distributions=param_grid, n_iter=10,
        scoring='accuracy', cv=5, n_jobs=-1)
    random_search.fit(X, y)

    # Best hyperparameters found
    best_params = random_search.best_params_

    # Print best hyperparameters and best cross-validation score
    print("✅ Best hyperparams found")
    print("Best Hyperparameters:", best_params)
    print("Best Cross-validation Score:", random_search.best_score_)

    # Return the best parameters
    return best_params

def train_model_svm(X, y, probability=True, best_params=None):
    if best_params is None:
        # If best_params is not provided, train the model with default parameters
        model = SVC(kernel=KERNEL_DEFAULT, gamma=GAMMA_DEFAULT, C=C_DEFAULT, probability=probability)
    else:
        # If best_params is provided, extract kernel and penalty_c from it
        kernel_name = best_params['kernel']
        penalty_c = best_params['C']
        model = SVC(kernel=kernel_name, C=penalty_c, probability=probability)

    model.fit(X, y)
    print("✅ Model fit \n")
    return model

def evaluate_model_svm(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy



def randomized_search_model_knn(X, y):
    # Define the hyperparameter grid
    param_grid = {
    'n_neighbors':  N_NEIGHBORS,  # Penalty parameter C (regularization parameter)
    'leaf_size': LEAF_SIZE,
    'weights': ['uniform', 'distance']
    }

    # Create an KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Perform random search cross-validation
    random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_grid, n_iter=100,  scoring='accuracy', cv=5, n_jobs=-1)
    random_search.fit(X, y)

    # Best hyperparameters found
    best_params = random_search.best_params_

    # Print best hyperparameters and best cross-validation score
    print("Best Hyperparameters:", best_params)
    print("Best Cross-validation Score:", random_search.best_score_)

    # Return the best parameters
    return best_params

def train_model_knn(X, y, best_params=None):
    if best_params is None:
        # If best_params is not provided, train the model with default parameters
        model = KNeighborsClassifier()
    else:
        # If best_params is provided, extract kernel and penalty_c from it
        n_neighbors = best_params['n_neighbors']
        leaf_size = best_params['leaf_size']
        model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)

    model.fit(X, y)
    return model

def evaluate_model_knn(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy

if __name__ == '__main__':
    features_path= os.path.join(os.path.dirname(__file__), "../../raw_data/features_1000sample_400min_600cutoff.csv")
    features_df = pd.read_csv(features_path)
    target_path= os.path.join(os.path.dirname(__file__), "../../raw_data/target_1000sample_400min_600cutoff.csv")
    target_df = pd.read_csv(target_path)
    features_df.drop(columns = ['Unnamed: 0'], inplace = True)
    target_df.drop(columns = ['Unnamed: 0'], inplace = True)
    train_model_knn(features_df, target_df)
