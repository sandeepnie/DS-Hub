# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline thact cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. Go to http://0.0.0.0:3001/


### Project Overview
In the Project Workspace/data, we have a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

THis project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


### Project Components
There are three components of this project.

1. ETL Pipeline
    - In a Python script, process_data.py, data cleaning pipeline is coded with below tasks: 
        - Loads the messages and categories datasets
        - Merges the two datasets
        - Cleans the data
        - Stores it in a SQLite database
        
2. ML Pipeline
    - In a Python script, train_classifier.py, a machine learning pipeline coded with below tasks that:
        - Loads data from the SQLite database
        - Splits the dataset into training and test sets
        - Builds a text processing and machine learning pipeline
        - Trains and tunes a model using GridSearchCV
        - Outputs results on the test set
        - Exports the final model as a pickle file
3. Flask Web App
    - Web app is a place where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

    - Below are a few screenshots of the web app.
    
        ![](disaster-response-project1.png)
        ![](disaster-response-project2.png)
