FROM jupyter/scipy-notebook:latest
WORKDIR /recommender
COPY requirements.txt requirements.txt
USER root
RUN apt -y update && apt install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt
