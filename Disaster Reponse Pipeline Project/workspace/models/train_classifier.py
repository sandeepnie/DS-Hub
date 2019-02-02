import sys
import pandas as pd
import numpy as np
import sqlalchemy

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    
    """
    This function will read the data from the excel file and load values into database
    Input: 
        database_filepath: path to database
    Output:
        X : features
        y: labels
        category_names: name of the label columns
    """    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', con= engine)
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis = 1) 
    category_names = y.columns

    return X,y, category_names


def tokenize(text):
    """
    This function takes string as a parameter and returns clean tokens
    Input:
        text: string input
    Output:
        clean_tokens: returns word tokens    
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
        This model does not expect any argument and returns the model
    """
    
    pipeline = Pipeline([
        ('vect' , CountVectorizer()),
        ('tfidf' , TfidfTransformer()),
        ('clf' , RandomForestClassifier())
    ])
    
    parameters = {'clf__max_depth': [10, 20, None],
              'clf__min_samples_leaf': [1, 2, 4],
              'clf__min_samples_split': [2, 5, 10],
              'clf__n_estimators': [10, 20, 40]}

    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = 'f1_micro', n_jobs = -1) #n_jobs = -1 means use all processors
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This model takes the model, features, labels and catrogy names as input and evaluates the model
    
    Input:
        model: trained model
        X_test: test features
        Y_test: test labels
        category_names: String array of category names
    Output:
        N/A
    """
    
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names= category_names))
    print("**** Accuracy scores for each category *****\n")
    for i in range(36):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    """
    Save the model to a Python pickle
    Input:
        model: Trained model
        model_filepath: Path where to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()