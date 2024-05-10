# Is this the best one to use? For mac?
FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt
COPY polclassifier /polclassifier

RUN pip install -r requirements.txt

CMD uvicorn polclassifier.api.fast:app --host 0.0.0.0 --port $PORT

# CLI code needed
### GAR_IMAGE=polclassifier
### docker build --tag=$GAR_IMAGE:dev .
### docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev -> include sh on the end to enter shell to test ls and pip list
