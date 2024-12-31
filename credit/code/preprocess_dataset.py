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
    X_train = train_data.drop(columns=['class'])
    y_train = train_data['class']
    
    # Split into features and target
    X_test = test_data.drop(columns=['class'])
    y_test = test_data['class']

    # Initialize HyperTransformer
    ht = HyperTransformer()

    # Detect initial configuration (optional for debugging)
    detected_datatype = ht.detect_initial_config(data=X_train)
    print("Detected Data Types:", ht.get_config())

    # 'Attribute2', 'Attribute5',  'Attribute8', 'Attribute11', 'Attribute16', 'Attribute18', 'Attribute13'
    # 'Attribute1', 'Attribute3', 'Attribute4', 'Attribute6', 'Attribute7', 'Attribute9', 'Attribute10', 'Attribute12', 'Attribute14', 'Attribute15', 'Attribute17', 'Attribute19', 'Attribute20'
    
    # ['Attribute2', 'Attribute5', 'Attribute8', 'Attribute11', 'Attribute16', 'Attribute18', 'Attribute13']
    
    # Define custom configuration
    ht.set_config(
        {
        "sdtypes": {
            # Numerical columns
            "Attribute2": "numerical",
            "Attribute5": "numerical",
            "Attribute8": "numerical",
            "Attribute11": "numerical",
            "Attribute16": "numerical",
            "Attribute18": "numerical",
            "Attribute13": "numerical",         # continous age
            
            # Categorical columns
            "Attribute1": "categorical",
            "Attribute3": "categorical",
            "Attribute4": "categorical",
            "Attribute6": "categorical",
            "Attribute7": "categorical",
            "Attribute9": "categorical",        # discritized gender
            "Attribute10": "categorical",
            "Attribute12": "categorical",
            "Attribute14": "categorical",
            "Attribute15": "categorical",
            "Attribute17": "categorical",
            "Attribute19": "categorical",
            "Attribute20": "categorical"
        },
        "transformers": {
            # Numerical columns with GaussianNormalizer
            "Attribute2": GaussianNormalizer(),
            "Attribute5": GaussianNormalizer(),
            "Attribute8": GaussianNormalizer(),
            "Attribute11": GaussianNormalizer(),
            "Attribute16": GaussianNormalizer(),
            "Attribute18": GaussianNormalizer(),
            "Attribute13": GaussianNormalizer(),
            
            # Categorical columns with LabelEncoder
            "Attribute1": OneHotEncoder(),
            "Attribute3": OneHotEncoder(),
            "Attribute4": OneHotEncoder(),
            "Attribute6": OneHotEncoder(),
            "Attribute7": OneHotEncoder(),
            "Attribute9": OneHotEncoder(),
            "Attribute10": OneHotEncoder(),
            "Attribute12": OneHotEncoder(),
            "Attribute14": OneHotEncoder(),
            "Attribute15": OneHotEncoder(),
            "Attribute17": OneHotEncoder(),
            "Attribute19": OneHotEncoder(),
            "Attribute20": OneHotEncoder()
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
    
    train_file_path = "../dataset/credit_train.csv"
    test_file_path = "../dataset/credit_train.csv"
    models_output_dir = os.path.join(base_dir, "models")
    X_train_transformed, X_test_transformed  = preprocess_data(models_output_dir, train_file_path, test_file_path)
    print(f"X_train_transformed's shape: {X_train_transformed.shape} | X_train_transformed: {X_train_transformed[0:5]}")
    print("Standalone script execution complete.")