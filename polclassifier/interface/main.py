import os
from colorama import Fore, Style
import pandas as pd
pd.options.mode.chained_assignment = None

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split

from polclassifier.ml_logic.preprocessing import *
from polclassifier.ml_logic.models import *
from polclassifier.ml_logic.registry import *
from polclassifier.params import *

import shap
shap.initjs();

def preprocess(reprocess_by_default=False):

    print(Fore.MAGENTA + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)

    X_path = os.path.join(
        LOCAL_PATH, "processed_data",
        f"features_{SAMPLE_SIZE}sample_{MIN_WORD_COUNT}min_{MAX_WORD_COUNT}cutoff_{VECT_METHOD}.csv")
    y_path = os.path.join(
        LOCAL_PATH, "processed_data",
        f"target_{SAMPLE_SIZE}sample_{MIN_WORD_COUNT}min_{MAX_WORD_COUNT}cutoff_{VECT_METHOD}.csv")

    # Check cache
    # if there, load from there
    if os.path.isfile(X_path) and os.path.isfile(y_path) and not reprocess_by_default:
        print("✅ X and y loaded in from cache \n")
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
    # if not, do proper preprocessing
    else:
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
            vect_method=VECT_METHOD, local_path=LOCAL_PATH)

        print("✅ X and y preprocessed \n")

        save_processed_to_cache(
            pd.DataFrame(X), y, local_path=LOCAL_PATH, sample_size=SAMPLE_SIZE,
            min_word_count=MIN_WORD_COUNT, max_word_count=MAX_WORD_COUNT,
            vect_method=VECT_METHOD)

        print("✅ X and y saved to cache \n")

    return X, y

def train_evaluate_model_svm(split_ratio: float = 0.2, perform_search: bool = False):
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Retrieve data
    X, y = preprocess()

    # Extract series from y if saved as a DataFrame
    if len(y.shape) > 1:
        y = y["party"]

    # print(f"y shape: {y.shape}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42, stratify=y)

    print("Data split in to test and train")

    if perform_search:
        print("Searching for the best parameters ...")
        # Search for best parameters
        best_params = randomized_search_model_svm(X_train, y_train)
    else:
        # Use default parameters
        best_params = None

    # Train model using `models.py`
    print("Training SVM model")
    model = train_model_svm(X_train, y_train, probability=True, best_params=best_params)

    print("✅ Model trained \n")

    # Evaluate model using `models.py
    accuracy = evaluate_model_svm(model=model, X=X_test, y=y_test)

    print(f"Model accuracy: {accuracy}")

    # Save model weight on the hard drive (and optionally on GCS too!)
    print("Saving model...")
    save_model_sklearn(model=model)

    return accuracy



def pred_sklearn(speech: str = None) -> np.ndarray:

    """ Let's make a prediction using the latest ML model """

    # Create X_pred dataframe consisting of speech text and word count
    word_n_full = len(speech.strip().split())

    X_pred = pd.DataFrame(dict(
        text=[speech],
        word_n_full=[word_n_full],
    ))

    print("✅ Input string converted to dataframe, now preprocessing...\n")

    # Preprocess the input data
    X_processed = preprocess_text_col(X_pred)

    # Vectorise the processed text... HOW?

    if VECT_METHOD=="tfidf":

        tf_idf_vectorizer = load_vectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)

        X_vectorized = tf_idf_vectorizer.transform(X_processed).toarray()

    print("... and vectorizing! ✅ \n")

    # Load model functionality specific to ML models
    model = load_model_sklearn()
    assert model is not None

    y_pred = model.predict(X_vectorized)

    print(f"✅ And the winner is ... {y_pred}")

    # Predict probabilities if the model supports it
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_vectorized)
        print(f"✅ Probability estimates: {y_prob}")
        y_prob = np.max(y_prob[0])

    else:
        print("❌ Model does not support probability estimates.")
        y_prob = None

    return y_pred, y_prob



def visualise_pred(speech: str = None):

    # Create X_pred dataframe consisting of speech text and word count
    word_n_full = len(speech.strip().split())

    X_pred = pd.DataFrame(dict(
        text=[speech],
        word_n_full=[word_n_full],
    ))

    # Preprocess the input data
    X_processed = preprocess_text_col(X_pred)

    # Load model functionality specific to ML models
    model = load_model_sklearn()
    assert model is not None

    tf_idf_vectorizer = load_vectorizer(min_df=MIN_DF, max_df=MAX_DF, max_features=MAX_FEATURES)

    def make_predictions(X_batch_text):
        X_batch = tf_idf_vectorizer.transform(X_batch_text).toarray()
        preds = model.predict_proba(X_batch)
        return preds

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(make_predictions, masker=masker, output_names=["Con", "DUP", "Lab", "LibDem", "PlaidCymru", "SNP", "UUP"])

    print("✅ Shapley explainer trained on model...")

    # Create shapley values on the input string and save plot to registry
    shap_values = explainer(X_processed)

    save_shapley_plot(shap_values)



def train_evaluate_model_knn(split_ratio: float = 0.2, perform_search: bool = False):
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Retrieve data
    X, y = preprocess()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    if perform_search:
        # Search for best parameters
        best_params = randomized_search_model_knn(X_train, y_train)
    else:
        # Use default parameters
        best_params = {'weights': WEIGHTS_DEFAULT, 'n_neighbors': N_NEIGHBORS_DEFAULT, 'leaf_size': LEAF_SIZE_DEFAULT}

    # Train model using `models.py`
    model = train_model_knn(X_train, y_train, best_params=best_params)

    # Evaluate model using `models.py
    accuracy = evaluate_model_knn(model=model, X=X_test, y=y_test)

    save_model_sklearn(model=model)

    return model, accuracy


def load_speeches(min_word_count=400, sample_size=1000, parties_to_exclude=[], speeches_per_party = 20):

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

    df = pd.concat(smaller_data_test)

    def tokenize_if_needed(text):
        if isinstance(text, list):
            return text  # Already tokenized
        return text.split()  # Tokenize by splitting

    # Apply the tokenization function to ensure all texts are tokenized
    df['text'] = df['text'].apply(tokenize_if_needed)

    def cut_middle_200(words):
        if len(words) <= 200:
            return " ".join(words)  # Return the original text if it's shorter than 200 words
        else:
            start_index = (len(words) // 2) - 100  # Calculate the start index of the middle 200 words
            # Ensure the start index is non-negative
            start_index = max(start_index, 0)
            return "..." + " ".join(words[start_index:start_index + 200]) + "..."  # Add "..." around the middle 200 words

    df['sample_text'] = df['text'].apply(cut_middle_200)

    path = os.path.join(LOCAL_PATH, "processed_data", "smaller_data_sample_text.csv")
    df.to_csv(path, index=False)

    return df

if __name__ == '__main__':
    train_evaluate_model_knn()
    train_evaluate_model_svm()
    # load_speeches()
    # create_shapley_explainer()
    load_speeches()
