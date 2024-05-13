import pandas as pd
import numpy as np


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

from sklearn.svm import SVC

from colorama import Fore, Style
import joblib

from polclassifier.params import *

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

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
    print("✅ Model fit \n")
    return model

def evaluate_model_svm(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy

def initialize_model_lstm():
    """Initialize Recurrent Neural Network with LSTM
    """



    print("✅ Model initialized")

    return model

def compile_model_lstm(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("✅ Model compiled")

    return model

def train_model_lstm(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=32,
        patience=15,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with max accuracy: {round(np.max(history.history['accuracy']), 2)}")

    return model, history
