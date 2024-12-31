import os
import random
import numpy as np
import torch
import pandas as pd
from rdt import HyperTransformer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from datetime import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = ""
current_timestamp = datetime.now().strftime("%Y-%m-%d_%H")  # Format: YYYY-MM-DD_HH (24-hour format)


def initialize_seed_from_env():
    """
    Initialize the seed for all libraries using the seed stored in the GLOBAL_SEED environment variable.
    """
    seed = int(os.getenv("GLOBAL_SEED", 42))  # Default to 42 if not set
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for Pandas
    pd.core.common.random_state(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    print(f"Seed initialized to: {seed}")



def export_loss(synthesizer, output_file_name, embedding_dim):
    
    """
    Export summary loss metrics to a CSV file.
    Args:
        synthesizer: The synthesizer object containing loss values.
        output_file_name (str): The base name for the output CSV file.
        embedding_dim (int): The embedding dimension to include in the file name.
    Returns:
        None
    """

    # Retrieve loss values and calculate metrics
    loss_values = synthesizer.get_loss_values()
    loss_df = pd.DataFrame(loss_values)
    epoch_loss = loss_df.groupby("Epoch")["Loss"].mean()

    # Calculate metrics
    avg_loss_last_epoch = epoch_loss.iloc[-1]
    avg_loss_last_10_epochs = epoch_loss.tail(10).mean()
    avg_loss_last_20_epochs = epoch_loss.tail(20).mean()

    # Save metrics to a CSV
    metrics_df = pd.DataFrame({
        "Embedding_Dim": [embedding_dim],
        "Avg_Loss_Last_Epoch": [avg_loss_last_epoch],
        "Avg_Loss_Last_10_Epochs": [avg_loss_last_10_epochs],
        "Avg_Loss_Last_20_Epochs": [avg_loss_last_20_epochs],
    })
    
    metrics_df.to_csv(output_file_name, index=False)
    
    # Print confirmation
    print(f"Summary metrics saved to {output_file_name}")



def train_vae(X_train_transformed, embedding_dim, output_dir):
    """
    Train a VAE model using transformed data.
    Args:
        X_train_transformed (pd.DataFrame): Preprocessed training dataset.
        embedding_dim (int): Embedding dimension for the VAE.
        output_dir (str): Directory to save the trained models.
    Returns:
        None
    """
    initialize_seed_from_env()
    
    if X_train_transformed is None:
        raise ValueError("X_train_transformed must be provided for training.")

    # Step 1: Detect metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=X_train_transformed)
    
    # Step 2: Initialize the TVAESynthesizer
    synthesizer = TVAESynthesizer(
        metadata=metadata,
        batch_size=32,
        enforce_min_max_values=True,
        enforce_rounding=False,
        epochs=500,
        verbose=True,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        embedding_dim=embedding_dim
    )
    
    # Step 3: Fit the synthesizer to the transformed data
    synthesizer.fit(X_train_transformed)

    # Step 4: Save the trained synthesizer model
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, f'{embedding_dim}_tvae_COMPAS_synthesizer_model_{current_timestamp}.pkl')
    synthesizer.save(model_save_path)
    
    loss_save_path = os.path.join(output_dir, f'{embedding_dim}_tvae_COMPAS_synthesizer_model_loss_summary_metrics_{current_timestamp}.csv')
    export_loss(synthesizer, loss_save_path, embedding_dim)
    print(f"Model saved at: {model_save_path}")
    print(f"\n current_timestamp: {current_timestamp}")