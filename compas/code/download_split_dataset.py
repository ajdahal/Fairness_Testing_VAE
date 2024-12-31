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


def default_preprocessing(df):          # this is from AI360
    """
    Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    return df[
        (df.days_b_screening_arrest <= 30)
        & (df.days_b_screening_arrest >= -30)
        & (df.is_recid != -1)
        & (df.c_charge_degree != 'O')
        & (df.score_text != 'N/A')
    ]


def process_and_split_data(dataset_dir):
    """
    Fetch, preprocess, and split the compas dataset.
    Args:
        dataset_dir (str): Path to the directory where processed files will be saved.
    Returns:
        pd.DataFrame, pd.DataFrame: The training and testing datasets.
    """
    initialize_seed_from_env()
    
    # URL of the raw CSV file
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

    # File name to save the file locally
    file_name = "compas-scores-two-years.csv"

    # Send GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write content to a file
        with open(file_name, "wb") as file:
            file.write(response.content)
        print(f"File saved as {file_name}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    
    # Extract features and targets
    data = pd.read_csv(file_name)
    
    # Apply default Preprocessing
    data = default_preprocessing(data)
    
    # Columns to retain
    # sex,age,age_cat,race,juv_fel_count,juv_misd_count,juv_other_count,priors_count,c_charge_degree,two_year_recid   --> c_charge_desc is removed
    columns_to_keep = ['sex','age','age_cat','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree','two_year_recid']
    data = data[columns_to_keep]
    
    # Remove rows with variations of ' ?'
    def clean_question_mark(value):
        if isinstance(value, str) and value.strip() == '?':
            return None
        return value
    
    data = data.applymap(clean_question_mark).dropna()

    # Save all data points to compas_all_data.csv
    os.makedirs(dataset_dir, exist_ok=True)
    all_data_file_path = os.path.join(dataset_dir, "compas_all_data.csv")
    data.to_csv(all_data_file_path, index=False)
    print(f"All data saved to: {all_data_file_path}, Number of instances: {len(data)}")

    # Split dataset into training and testing sets
    X_actual = data.drop(columns=['two_year_recid'])
    y_actual = data['two_year_recid']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_actual, y_actual, test_size=0.20, random_state=42, stratify=y_actual)
    
    # Combine X_train and y_train, X_test and y_test
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Define train and test file paths
    train_file_path = os.path.join(dataset_dir, "compas_train.csv")
    test_file_path = os.path.join(dataset_dir, "compas_test.csv")

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