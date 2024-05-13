from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from polclassifier.interface.main import preprocess


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
