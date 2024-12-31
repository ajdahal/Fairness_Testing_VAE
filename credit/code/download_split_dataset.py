import os
import random
import numpy as np
import torch
import requests
import pandas as pd
from rdt import HyperTransformer
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo



def initialize_seed_from_env():
    """
    Initialize the seed for all libraries using the seed stored in the GLOBAL_SEED environment variable.
    """
    seed = int(os.getenv("GLOBAL_SEED", 42))  # Default to 42 if not set
    # Set the seed for Pandas
    pd.core.common.random_state(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    print(f"Seed initialized to: {seed}")


def map_personal_status_and_sex(df, column_name):
    """
    Map Attribute 9 to 'male' or 'female' based on the given codes.
    """
    mapping = {
        'A91': 'Male',
        'A92': 'Female',
        'A93': 'Male',
        'A94': 'Male',
        'A95': 'Female'
    }
    df[column_name] = df[column_name].map(mapping)
    return df


def process_and_split_data(dataset_dir):
    """
    Fetch, preprocess, and split the Credit dataset.
    Args:
        dataset_dir (str): Path to the directory where processed files will be saved.
    Returns:
        pd.DataFrame, pd.DataFrame: The training and testing datasets.
    """
    initialize_seed_from_env()
    
    # fetch dataset 
    statlog_german_credit_data = fetch_ucirepo(id=144) 
  
    # data (as pandas dataframes) 
    X = statlog_german_credit_data.data.features 
    y = statlog_german_credit_data.data.targets 
  
  
    print(f"\n X: {X[0:5]}")
    print(f"\n y: {y[0:5]}")
    
    # metadata 
    # print(statlog_german_credit_data.metadata) 
    
    # variable information 
    # print(statlog_german_credit_data.variables) 
    
    data = pd.concat([X, y], axis=1)
     
    columns_to_keep = [
        'Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5',
        'Attribute6', 'Attribute7', 'Attribute8', 'Attribute9', 'Attribute10', 
        'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15', 
        'Attribute16', 'Attribute17', 'Attribute18', 'Attribute19', 'Attribute20',
        'class'
        
    ]
    data = data[columns_to_keep]

    # Remove rows with variations of ' ?'
    def clean_question_mark(value):
        if isinstance(value, str) and value.strip() == '?':
            return None
        return value

    data = data.applymap(clean_question_mark).dropna()
    
    os.makedirs(dataset_dir, exist_ok=True)
    all_data_file_path_before_mapping = os.path.join(dataset_dir, "credit_data_before_age_and_gender_mapping.csv")
    data.to_csv(all_data_file_path_before_mapping, index=False)
    print(f"All data before mapping saved to: {all_data_file_path_before_mapping}, Number of instances: {len(data)}")
    
    # Map Attribute 9 | Status to Sex
    data = map_personal_status_and_sex(data, column_name='Attribute9')
    
    
    # Save all data points to credit_all_data.csv
    all_data_file_path = os.path.join(dataset_dir, "credit_all_data.csv")
    data.to_csv(all_data_file_path, index=False)
    print(f"All data saved to: {all_data_file_path}, Number of instances: {len(data)}")

    # Split dataset into training and testing sets
    X_actual = data.drop(columns=['class'])
    y_actual = data['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_actual, y_actual, test_size=0.20, random_state=42,stratify=y_actual)
    
    # Combine X_train and y_train, X_test and y_test
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Define train and test file paths
    train_file_path = os.path.join(dataset_dir, "credit_train.csv")
    test_file_path = os.path.join(dataset_dir, "credit_test.csv")
    
    # Save train and test datasets to CSV
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)
    
    print(f"Train data saved to: {train_file_path}, Number of instances: {len(train_data)}")
    print(f"Test data saved to: {test_file_path}, Number of instances: {len(test_data)}")
    return train_data, test_data


if __name__ == "__main__":
    # Standalone script execution
    initialize_seed_from_env()
    train_data, test_data = process_and_split_data(dataset_dir="../dataset/")
    print("Standalone script execution complete.")