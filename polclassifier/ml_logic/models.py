import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from polclassifier.params import *

def save_model(X, y, model, model_name):
    # Function for saving the models
    model.fit(X, y)
    joblib.dump(model, f'../models/{model_name}.pkl')

#HL Part
def randomized_search_model_svm(X, y):
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
    random_search = RandomizedSearchCV(svm_classifier, param_distributions=param_grid, n_iter=10, scoring='accuracy', cv=5, n_jobs=-1)
    random_search.fit(X, y)

    # Best hyperparameters found
    best_params = random_search.best_params_

    # Print best hyperparameters and best cross-validation score
    print("Best Hyperparameters:", best_params)
    print("Best Cross-validation Score:", random_search.best_score_)

    # Return the best parameters
    return best_params

def train_model_svm(X, y, best_params=None):
    if best_params is None:
        # If best_params is not provided, train the model with default parameters
        model = SVC(kernel=KERNEL_DEFAULT, gamma=GAMMA_DEFAULT, C=C_DEFAULT)
    else:
        # If best_params is provided, extract kernel and penalty_c from it
        kernel_name = best_params['kernel']
        penalty_c = best_params['C']
        model = SVC(kernel=kernel_name, C=penalty_c)

    model.fit(X, y)
    return model

def evaluate_model_svm(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy


def randomized_search_model_knn(X, y):
    # Define the hyperparameter grid
    param_grid = {
    'n_neighbors':  N_NEIGHBORS,  # Penalty parameter C (regularization parameter)
    'leaf_size': LEAF_SIZE
    }

    # Create an KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Perform random search cross-validation
    random_search = RandomizedSearchCV(knn_classifier, param_distributions=param_grid, n_iter=10, scoring='accuracy', cv=5, n_jobs=-1)
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
