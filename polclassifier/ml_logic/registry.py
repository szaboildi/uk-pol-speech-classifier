import os
import time
from colorama import Fore, Style


import joblib
import glob

from tensorflow import keras
from google.cloud import storage

from polclassifier.params import *


def save_model_sklearn(model = None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.pkl")
    joblib.dump(model, model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None


def save_model_keras(model: keras.Model = None) -> None:

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None


def load_model_keras() -> keras.Model:

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        
        except:
            
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    print("\n❌ Model target not recognised")
    
    return None



def load_model_sklearn():

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")


        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_path = os.path.join(LOCAL_REGISTRY_PATH, "models", most_recent_model_path_on_disk)
        latest_model = joblib.load(latest_path)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = joblib.load(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        
        except:
            
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    print("\n❌ Model target not recognised")
    
    return None


def save_vectorizer(vectorizer=None, min_df=5, max_df=0.85, max_features=10000) -> None:
    
    # Save vectorizer locally
    vect_path = os.path.join(LOCAL_REGISTRY_PATH, "vectorizers", f"{min_df}-{max_df}-{max_features}.pkl")
    joblib.dump(vectorizer, vect_path)

    print("✅ Vectorizer saved locally")    
    
    if MODEL_TARGET == "gcs":

        vect_filename = vect_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"vectorizers/{vect_filename}")
        blob.upload_from_filename(vect_path)

        print("✅ Model saved to GCS")

        return None
    
    return None


def load_vectorizer(min_df=5, max_df=0.85, max_features=10000):
    
    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad vectorizer with params {min_df}, {max_df}, {max_features}" + Style.RESET_ALL)

        vect_path = os.path.join(LOCAL_REGISTRY_PATH, "vectorizers", f"{min_df}-{max_df}-{max_features}.pkl")
        vectorizer = joblib.load(vect_path)

        print("✅ Model loaded from local disk")

        return vectorizer

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad vectorizer with params {min_df}, {max_df}, {max_features} from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="vectorizers"))

        try:
            vect_name = f"{min_df}-{max_df}-{max_features}.pkl"
            vect_blob = next((blob for blob in blobs if blob.name == vect_name), None)
            vect_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, vect_blob.name)
            vect_blob.download_to_filename(vect_path_to_save)

            vectorizer = joblib.load(vect_path_to_save)

            print("✅ Vectorizer downloaded from cloud storage")

            return vectorizer
        
        except:
            
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    print("\n❌ Model target not recognised")
    
    return None

