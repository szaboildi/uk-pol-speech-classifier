import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from polclassifier.interface.main import pred


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
    
    word_n_full = len(speech.strip().split())
    X_pred = pd.DataFrame(dict(
        text=[speech],
        word_n_full=[word_n_full]
    ))
    
    y_pred = pred(X_pred)[0][0]
    
    return dict(party = y_pred)

@app.get("/")
def root():
    return {"greeting": "Is this thing on?"}
 