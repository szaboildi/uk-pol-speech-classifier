# To build & run locally
### GAR_IMAGE=polclassifier
### docker build --tag=$GAR_IMAGE:dev .
### docker run -it -e PORT=8000 -p 8000:8000 --env-file .env $GAR_IMAGE:dev -> include sh on the end to enter shell to test ls and pip list

FROM tensorflow/tensorflow:2.18.0

WORKDIR /prod

# First, pip install dependencies
COPY requirements.txt requirements.txt
COPY processed_data/smaller_data_sample_text.csv smaller_data_sample_text.csv
RUN pip install --no-cache-dir -r requirements.txt

# Then only, install taxifare!
COPY polclassifier polclassifier
COPY setup.py setup.py
COPY training_outputs training_outputs
RUN pip install .

CMD uvicorn polclassifier.api.fast:app --host 0.0.0.0 --port $PORT
