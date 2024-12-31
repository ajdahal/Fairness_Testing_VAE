import os
import random
import numpy as np
import torch
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



def process_and_split_data(dataset_dir):
    """
    Fetch, preprocess, and split the Adult dataset.
    Args:
        dataset_dir (str): Path to the directory where processed files will be saved.
    Returns:
        pd.DataFrame, pd.DataFrame: The training and testing datasets.
    """
    initialize_seed_from_env()
    
    # Fetch dataset from UCI repository
    adult = fetch_ucirepo(id=2)

    # Extract features and targets
    X = adult.data.features
    y = adult.data.targets

    # Combine features and target into one DataFrame
    data = pd.concat([X, y], axis=1)
    
    print(data.columns)
    
    # Columns to retain
    columns_to_keep = ['age', 'workclass', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'income']                                                                    # fnlwgt is removed
    
    data = data[columns_to_keep]
    
    # Remove rows with variations of ' ?'
    def clean_question_mark(value):
        if isinstance(value, str) and value.strip() == '?':
            return None
        return value

    data = data.applymap(clean_question_mark).dropna()

    # Normalize and map target column to binary integer values
    def normalize_income(value):
        if isinstance(value, str):
            value = value.strip().lower() 
            if value == '>50k':
                return 1
            elif value == '<=50k':
                return 0
        return None
    
    data['income'] = data['income'].map(normalize_income)
    data = data.dropna(subset=['income']).astype({'income': 'int'})

    # Save all data points to adult_all_data.csv
    os.makedirs(dataset_dir, exist_ok=True)
    all_data_file_path = os.path.join(dataset_dir, "adult_all_data.csv")
    data.to_csv(all_data_file_path, index=False)
    print(f"All data saved to: {all_data_file_path}, Number of instances: {len(data)}")

    # Split dataset into training and testing sets
    X_actual = data.drop(columns=['income'])
    y_actual = data['income']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_actual, y_actual, test_size=0.20, random_state=42,stratify=y_actual)

    # Combine X_train and y_train, X_test and y_test
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Define train and test file paths
    train_file_path = os.path.join(dataset_dir, "adult_train.csv")
    test_file_path = os.path.join(dataset_dir, "adult_test.csv")
    
    # Save train and test datasets to CSV
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)

    print(f"Train data saved to: {train_file_path}, Number of instances: {len(train_data)}")
    print(f"Test data saved to: {test_file_path}, Number of instances: {len(test_data)}")
    return train_data, test_data


if __name__ == "__main__":
    # Standalone script execution
    initialize_seed_from_env()
    train_data, test_data = process_and_split_data(dataset_dir = "../dataset/")
    print("Standalone script execution complete.")