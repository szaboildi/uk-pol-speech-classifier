import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


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
    model.fit(X, y)
    joblib.dump(model, f'../models/{model_name}.pkl')

model = KNeighborsClassifier()
save_model(features_df, target_df, model, 'KNeighborsClassifier')

model = LogisticRegression()
save_model(features_df, target_df, model, 'LogisticRegression')

model = SGDClassifier()
save_model(features_df, target_df, model, 'SGDClassifier')
