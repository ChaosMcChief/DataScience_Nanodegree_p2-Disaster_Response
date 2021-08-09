import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
    Loads the specified csv-files of the messages and categories and 
    merges them together.
    
    Input:
    messages_filepath: str      -   Path to the messages-csv-file
    categories_filepath: str    -   Path to the categories-csv-file
    
    Output:
    df: pd.DataFrame            -   Output DataFrame with the loaded data
    """

    # Loading the csv-files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merging the two dataframes into one
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    Cleaning the dataframe with the following steps:
        1) splitting the categories in one column per category
        2) extract the name of each category and rename the columns
        3) converting the category-values into numerical values
        4) replace the 'categories' column in the original DataFrame
        5) drop duplicates in the dataframe
        
    Input:
    df: pd.DataFrame    -   Dataframe containing the messages and categories (output from load_data)
    
    Output:
    df: pd.DataFrame    -   Dataframe with the cleaned data
    """
    
    # 1) Splitting the categories
    categories = df['categories']
    categories = categories.str.split(";", expand=True)

    # 2) extract the names of each category
    row = categories.iloc[0,:]
    category_colnames = row.str.split("-", expand=True).iloc[:,0].values
    categories.columns = category_colnames 
    
    # 3) converting the category-values into numerical values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    
    # Replacing any 2 in the related category with a 1. Since we don't know anything about that class, I'll take the class with
    # the most entries. This should not be changing the results much.
    categories['related']=categories['related'].map(lambda x: 1 if x == 2 else x)
    
    # 4) replace the 'categories' column in the original DataFrame
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat((df, categories), axis=1)
    
    # 5) drop duplicates in the dataframe
    df.drop_duplicates(inplace=True)
    
    return df
    
def save_data(df, database_filename):
    """
    Saves the data stored in df to a sqlite database.
    
    Input:
    df: pd.DataFrame        -   Dataframe containing the messages and categories 
    database_filename: str  -   Filename in which the database should be stored.
    
    Output:
    None
    """
    
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("Data_Table", engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    # Test-Call: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse
    