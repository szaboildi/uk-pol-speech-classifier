import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from polclassifier.interface.main import pred_sklearn, visualise_pred
from polclassifier.params import *
import random
from fastapi.responses import FileResponse


app = FastAPI()

# What is this step??? -> "Allowing all middleware is optional, but good practice for dev purposes"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(speech: str) -> dict:

    y_pred, y_prob = pred_sklearn(speech)
    y_pred = y_pred.tolist()

    return dict(party=y_pred[0], probability=y_prob)

# Read in smaller dataset
data = pd.read_csv("smaller_data_sample_text.csv")


# Define endpoint for speech selection
@app.get('/speech')
def get_speech(party: str):

    # Filter data based on party
    party_data = data[data['party'] == party].reset_index(drop=True)

    # Check if there are speeches available for the selected party
    if party_data.empty:
        return {"error": "No speeches found for the selected party."}

    # Select a random speech from the filtered data
    selected_speech = random.choice(party_data['sample_text'])
    return dict(speech = selected_speech)



@app.get("/visualisation")
def visualise_predict(speech: str):

    visualise_pred(speech)
    html_file = os.path.join("training_outputs", "text_plot", "latest_plot.html")
    return FileResponse(html_file)



@app.get("/")
def root():
    return dict(greeting = "Is this thing on?")
