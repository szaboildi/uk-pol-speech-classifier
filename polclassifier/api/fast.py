import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from polclassifier.interface.main import pred_sklearn
from polclassifier.params import *
import random


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

    y_pred = pred_sklearn(speech)[0]

    return dict(party = y_pred)

# Load data
data = pd.read_csv("smaller_data_test.csv")

# Define endpoint for speech selection
@app.get('/speech')
def get_speech(party: str):

    # Filter data based on party
    party_data = data[data['party'] == party]

    # Check if there are speeches available for the selected party
    if party_data.empty:
        return {"error": "No speeches found for the selected party."}

    # Select a random speech from the filtered data
    selected_speech = random.choice(party_data['text'])
    return dict(speech = selected_speech)

@app.get("/")
def root():
    return dict(greeting = "Is this thing on?")
