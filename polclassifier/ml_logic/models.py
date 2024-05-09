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


# Read CSV files with correct headers or without headers
features_df = pd.read_csv('../../raw_data/features_1000sample_300min_500cutoff.csv', header=None)
target_df = pd.read_csv('../../raw_data/target_1000sample_300min_500cutoff.csv', header=None)

# Merge features and target
data_df = pd.concat([features_df, target_df], axis=1)

# Preprocess data (handle missing values, encode categorical variables, etc.)

# Assuming the correct column name for the target variable is 'target'
target_column_name = 'target'

# Split data into features and target
X = data_df.drop(target_column_name, axis=1)
y = data_df[target_column_name]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
import joblib
joblib.dump(model, 'logistic_regression_model.pkl')

#RB Part
# KNN Model==================================================================

features_df = pd.read_csv('../raw_data/features_1000sample_400min_600cutoff.csv')
target_df = pd.read_csv('../raw_data/target_1000sample_400min_600cutoff.csv')

features_df.drop(columns = ['Unnamed: 0'], inplace = True)
target_df.drop(columns = ['Unnamed: 0'], inplace = True)
features_df.info()
target_df.info()

X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.30, random_state=42, stratify = target_df)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

model.score(X_test, y_test)

y_pred = model.predict(X_test)

print(classification_report(y_pred, y_test))

model.fit(features_df, target_df)
joblib.dump(model, '../models/KNN_model.pkl')

# Logistic Regression Model=====================================================
features_df = pd.read_csv('../raw_data/features_1000sample_400min_600cutoff.csv')
target_df = pd.read_csv('../raw_data/target_1000sample_400min_600cutoff.csv')

features_df.drop(columns = ['Unnamed: 0'], inplace = True)
target_df.drop(columns = ['Unnamed: 0'], inplace = True)
features_df.info()
target_df.info()

X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.30, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

model.score(X_test, y_test)

y_pred = model.predict(X_test)

print(classification_report(y_pred, y_test))
model.fit(features_df, target_df)
import joblib
joblib.dump(model, '../models/logistic_regression_model.pkl')

# SGDClassifier=====================================================

features_df = pd.read_csv('../raw_data/features_1000sample_400min_600cutoff.csv')
target_df = pd.read_csv('../raw_data/target_1000sample_400min_600cutoff.csv')

features_df.drop(columns = ['Unnamed: 0'], inplace = True)
target_df.drop(columns = ['Unnamed: 0'], inplace = True)
features_df.info()
target_df.info()

X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.30, random_state=42)

model = SGDClassifier()
model.fit(X_train, y_train)

model.score(X_test, y_test)

y_pred = model.predict(X_test)

print(classification_report(y_pred, y_test))

model.fit(features_df, target_df)
import joblib
joblib.dump(model, '../models/SGDClassifier_model.pkl')


def save_model(X, y, model, model_name):
    # Function for saving the models
    model.fit(X, y)
    joblib.dump(model, f'../models/{model_name}.pkl')

model = KNeighborsClassifier()
save_model(features_df, target_df, model, 'KNeighborsClassifier')

model = LogisticRegression()
save_model(features_df, target_df, model, 'LogisticRegression')

model = SGDClassifier()
save_model(features_df, target_df, model, 'SGDClassifier')

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
