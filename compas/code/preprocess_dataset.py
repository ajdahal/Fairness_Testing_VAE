import os
import random
import torch
import pickle
import numpy as np
import pandas as pd
from rdt import HyperTransformer
from rdt.transformers import GaussianNormalizer, OneHotEncoder
from sklearn.model_selection import train_test_split


def initialize_seed_from_env():
    """
    Initialize the seed for all libraries using the seed stored in the GLOBAL_SEED environment variable.
    """
    seed = int(os.getenv("GLOBAL_SEED", 42))  # Default to 42 if not set
    # Set the seed for Pandas
    pd.core.common.random_state(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for rdt HyperTransformer
    ht = HyperTransformer()
    ht.random_state = seed
    print(f"Seed initialized to: {seed}")



def preprocess_data(models_output_dir, train_file_path, test_file_path):
    """
    Preprocess the training data using HyperTransformer.
    Args:
        train_file_path (str): Path to the training CSV file.
    Returns:
        pd.DataFrame: Transformed X_train data.
    """
    initialize_seed_from_env()
    
    # Load training data
    train_data = pd.read_csv(train_file_path, index_col=None)
    test_data = pd.read_csv(test_file_path, index_col=None)

    # Split into features and target
    X_train = train_data.drop(columns=['two_year_recid'])
    y_train = train_data['two_year_recid']
    
    # Split into features and target
    X_test = test_data.drop(columns=['two_year_recid'])
    y_test = test_data['two_year_recid']
    
    # Initialize HyperTransformer
    ht = HyperTransformer()

    # Detect initial configuration (optional for debugging)
    detected_datatype = ht.detect_initial_config(data=X_train)
    print("Detected Data Types:", ht.get_config())

    print(f"\n X_train: {X_test[0:5]}")
    
    print("Columns in X_train:", X_train.columns.tolist())
    print("Data types in X_train:\n", X_train.dtypes)
    
    # sex,age,age_cat,race,juv_fel_count,juv_misd_count,juv_other_count,priors_count,c_charge_degree,c_charge_desc,two_year_recid
    
    # int64: age, juv_fel_count, juv_misd_count, juv_other_count, priors_count
    
    ht.set_config(
    {
    "sdtypes": {
        "age": "numerical",
        "sex": "categorical", # 
        "age_cat": "categorical", # 
        "race": "categorical", # 
        "juv_fel_count": "numerical",
        "juv_misd_count": "numerical",
        "juv_other_count": "numerical",
        "priors_count": "numerical",
        "c_charge_degree": "categorical"  #
    },
    "transformers": {
        "age": GaussianNormalizer(),
        "sex": OneHotEncoder(), #
        "age_cat": OneHotEncoder(), #
        "race": OneHotEncoder(),    #
        "juv_fel_count": GaussianNormalizer(),
        "juv_misd_count": GaussianNormalizer(),
        "juv_other_count": GaussianNormalizer(),
        "priors_count": GaussianNormalizer(),
        "c_charge_degree": OneHotEncoder()  #
    }
    })
    
    # Fit HyperTransformer on training data
    ht.fit(X_train)

    print(f"\n train_file_path: {train_file_path}")
    data_file_used = os.path.splitext(os.path.basename(train_file_path))[0]
    hypertransformer_file = f'{data_file_used}_tvae_hypertransformer.pkl'
    hypertransformer_file_path = os.path.join(models_output_dir, hypertransformer_file)
    with open(hypertransformer_file_path, 'wb') as file:
        pickle.dump(ht, file)
    
    # Transform train data
    X_train_transformed = ht.transform(X_train)
    X_test_transformed = ht.transform(X_test)
    
    # Print transformed data to verify
    print("Transformed X_train:")
    print(X_train_transformed.head())
    return X_train_transformed, X_test_transformed



if __name__ == "__main__":
    # Standalone script execution
    initialize_seed_from_env()
    
    train_file_path = "../dataset/compas_train.csv"
    test_file_path = "../dataset/compas_test.csv"
    models_output_dir = os.path.join(base_dir, "models")
    X_train_transformed, X_test_transformed  = preprocess_data(models_output_dir, train_file_path, test_file_path)
    print(f"X_train_transformed's shape: {X_train_transformed.shape} | X_train_transformed: {X_train_transformed[0:5]}")
    print("Standalone script execution complete.")