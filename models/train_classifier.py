# import librarier
import os
import sys
import re
import pickle

import pandas as pd
from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# download nltk-content
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Loades the data stored int the sqlite database into a DataFrame.
    
    Input:
    database_filepath: str  -   Path to the sqlite database from which the data should be loaded
    
    Output:
    df: pd.DataFrame        -   DataFrame with the loaded data
    """
    
    engine = create_engine(f"sqlite:///{database_filepath}")    
    df = pd.read_sql_table(f"Data_Table", engine)
    
    # dropping the 'child_alone'-category, since it has just zeros and the 'original'-column since we're only interested in the english text
    df.drop(['child_alone', 'original'], axis=1, inplace=True)
    
    # It seems that if a category has a nan-value, 
    # all of the other columns are nans as well, hence they can be dropped
    df.dropna(inplace=True)
    
    # Split into X and y
    X = df['message']
    y = df.iloc[:,3:]
    category_names = y.columns
    
    
    return X, y, category_names
    
def tokenize(text):
    """
    Processes the text by replacing any URLs, tokenizing, lemmatizing and removing stop words.
    
    Input:
    text: str  -   Raw input text
    
    Output:
    clean_tokens: list  -   List of tokens containing the processed text
    
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")
    
    # replacing urls
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenizing the text
    tokens = word_tokenize(text)

    # lemmatizing and removing stop words
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stop_words]
    
    return clean_tokens


def build_model():
    """
    Builds the pipeline and performs a GridSearch over specified parameters to improve the 
    model's performance.
    
    Input:
    None
    
    Output:
    model: sklearn  -   List of tokens containing the processed text
    
    """
    
    
    # defining the pipeline
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
    
    # defining the parameter for the grid search
    parameters = {'clf__estimator__max_features': ["auto", "log2"],
              'clf__estimator__n_estimators': [10, 50]}

    # perform the grid search
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)

    #return the model
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluating and printing a models performance on a given test-set by calculating the models
    accuracy, precision and recall.
    
    Input:
    model: sklearn model        -   model which should be evaluated
    X_test: np.array            -   Input data of the test set
    y_test: np.array            -   Label data of the test set
    category_names: list of str -   List of the category names
    
    Output:
    None
    
    """
    
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f"{col}: \n")
        print(classification_report(y_test.values[:,i] , y_pred[:,i]))



def save_model(model, model_filepath):
    """
    Saving the best model from the grid search to a pickle-file.
    
    Input:
    model: sklearn model    -   model which should be saved
    model_filepath: str     -   Filepath of the pickle-file to which the 
                                model should be saved
   
    Output:
    None
   """ 
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    print(sys.argv[1])
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
path = "/home/workspace/data/DisasterResponse.db" 

if __name__ == '__main__':
    #test-call python train_classifier.py /home/workspace/data/DisasterResponse.db model.pkl 
    main()