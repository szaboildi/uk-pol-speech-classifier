from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from polclassifier.interface.main import preprocess


def plot_confusion_matrix(model):
    # Load data
    X_test, y_test = preprocess()
    X_test = X_test["text"]
    y_test = y_test["party"]
    labels = list(y_test.unique())

    # Generate predictions
    y_pred = model.predict(X_test)

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
