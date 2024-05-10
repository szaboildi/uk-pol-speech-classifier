import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

import os
import pandas as pd
from colorama import Fore, Style

import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from nltk.corpus import stopwords


def clean_data(df, min_word_count=400, sample_size=1000, parties_to_exclude=[]):
    """
    Cleans the data: only parties with enough data, standardized text length, split
    into features (pd.Series of texts) and target (pd.Series)
    df: dataframe to preprocess
    min_word_count: minimum length for a speech to be included in the sampling
    sample_size: number of samples to draw per party
    """
    df = df[["speaker", "party", "text"]]

    # Filter for min word count
    df["word_n_full"] = df.apply(lambda row: len(row["text"].strip().split()), axis=1)
    df = df[df["word_n_full"] >= min_word_count]

    # Only select big enough parties
    n_speeches_by_party = df.groupby("party").size().reset_index(name="n_speeches").\
        sort_values("n_speeches", ascending=False).reset_index(drop=True)
    big_parties = n_speeches_by_party[n_speeches_by_party.n_speeches > sample_size]["party"].tolist()
    df = df[(df["party"].isin(big_parties)) & (~(df["party"].isin(parties_to_exclude)))]

    # Undersample
    df_undersampled = pd.DataFrame()

    for group_name, group_data in df.groupby('party'):
        sampled_data = group_data.sample(sample_size)
        df_undersampled = pd.concat([df_undersampled, sampled_data], axis=0)

    df_undersampled.reset_index(drop=True, inplace=True)

    return df_undersampled


def clean_text(text:str):
    # remove whitespace
    text = text.strip()
    # lowercase characters
    text = text.lower()
    # remove numbers
    text = "".join([l for l in text if not l.isdigit()])
    # remove punctuation
    text = "".join([l for l in text if l not in string.punctuation])

    # remove double spaces
    text = text.replace("  ", " ")

    # tokenize
    tokens = word_tokenize(text)

    # # remove stopwords - we're doing this in the
    # stop_words = set(stopwords.words("english"))
    # tokens = [w for w in tokens if w not in stop_words]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # join tokens
    cleaned_text = " ".join(lemmatized_tokens)

    return cleaned_text


def preprocess_text_col(
    df, max_word_count=600, extract_from="middle"):
    """
    df: dataframe to process
    max_word_count: amount of text to truncate long speeches to
    extract_from: where to get the truncated speech from for long speeches
        possible values: "start", "middle", "end"
        only "middle" is implemented so far
    min_df: hyperparameter for the TfidfVectorizer
    max_df: hyperparameter for the TfidfVectorizer
    max_features: hyperparameter for the TfidfVectorizer
    """
    # Truncating
    if extract_from == "middle":
        df["text"] = df.apply(
            lambda x: x.text if x.word_n_full <= max_word_count
            else " ".join(x.text.split()[
                (x.word_n_full//2)-(max_word_count//2):(x.word_n_full//2)+max_word_count//2]),
            axis=1)

    # Clean truncated text
    clean_texts = df["text"].apply(clean_text)

    return clean_texts


def preprocess_all(df, min_word_count=400, sample_size=1000, parties_to_exclude=[],
                   max_word_count=600, extract_from="middle",
                   min_df=5, max_df=0.85, max_features=10000, vect_method="tfidf", local_path=None):
    """Preprocess all data"""
    df = clean_data(df, min_word_count=min_word_count, sample_size=sample_size,
                    parties_to_exclude=parties_to_exclude)

    print("✅ Raw data cleaned \n")

    X = preprocess_text_col(
        df[["text", "word_n_full"]], max_word_count=max_word_count,
        extract_from=extract_from)
    y = df["party"]

    print("✅ Text preprocessed - X created \n")

    if vect_method=="tfidf":
        # Vectorizing
        tf_idf_vectorizer = TfidfVectorizer(
            min_df=min_df, max_df=max_df, max_features=max_features,
            stop_words="english")

        X = tf_idf_vectorizer.fit_transform(X).toarray()

        print("✅ X vectorized (TfIDf) \n")
        
    elif vect_method=="for_embed":
        codes = pd.DataFrame(list(enumerate(y.unique())))
        codes.rename(columns={0:"party_id", 1: "party_name"}, inplace=True)
        codes.to_csv(os.path.join(
            local_path, "processed_data",
            f"targetcodes_{sample_size}sample_{min_word_count}min_{max_word_count}cutoff_{vect_method}.csv"), 
                     index=False)

        y = pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(y.values.reshape(-1, 1)))

    return X, y


def save_processed_to_cache(
    X, y, local_path, sample_size=1000, min_word_count=400, max_word_count=600,
    vect_method="_"):
    """Caches X and y to local"""
    X_path = os.path.join(
        local_path, "processed_data",
        f"features_{sample_size}sample_{min_word_count}min_{max_word_count}cutoff_{vect_method}.csv")
    y_path = os.path.join(
        local_path, "processed_data",
        f"target_{sample_size}sample_{min_word_count}min_{max_word_count}cutoff_{vect_method}.csv")

    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)
    return


def embed_and_pad(X, embedding, stop_words=stop_words):
    X_embed = embed_sentences(X, embedding, stop_words)
    
    maxlen = max([len(x) for x in X_embed])
    X_pad = pad_sequences(X_embed, dtype='float32', padding='post', maxlen=maxlen)



# Function to convert a sentence (list of words) into a matrix representing the words in the embedding space
def embed_sentence_with_TF(sentence, embedding, stop_words=stop_words):
    embedded_sentence = []
    for word in sentence:
        if word in embedding and word not in stop_words:
            embedded_sentence.append(embedding[word])
        
    return np.array(embedded_sentence)

# Function that converts a list of sentences into a list of matrices
def embed_sentences(embedding, sentences):
    embed = []
    
    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(embedding, sentence.split())
        embed.append(embedded_sentence)
        
    return embed