## DataScience_Nanodegree_p2-Disaster_Response

## Intro
This repository refers to the second project of the Udacity Data Science Nanodegree in which we create a webpage in python with an underlying nlp-classifier. The model takes in a disaster textmessage (e.g.  tweet) and classifies the subject on which the message is about. This can help organize aid missions to be more effecitve.

## What's in here?
In this repo you'll find everything you need to set up the website:
 - Jupyter Notebooks explains the steps of data-processing and training the machine learning model more in depth
 - The python scripts to load, clean and store the training data in a sqlite database
 - The scripts to run a flas application containing the website. 

## Install
On top of a standard Anaconda-installment, you'll need the following packages:
 - flask
 - plotly
 - sqlalchemy
 - nltk

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
