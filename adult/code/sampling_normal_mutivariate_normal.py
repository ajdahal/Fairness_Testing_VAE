import os
import re
import torch
import pickle
import random
import numpy as np
import pandas as pd
import argparse
import operator
import itertools
from rdt import HyperTransformer
from lime.discretize import EntropyDiscretizer
from collections import defaultdict
from functools import reduce
from sdv.single_table import TVAESynthesizer
from testflows.combinatorics import Covering
from datetime import datetime

current_timestamp = datetime.now().strftime("%Y-%m-%d_%H")  # Format: YYYY-MM-DD_HH (24-hour format)
reconstructed_samples_path = "sampling_results"


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
    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)
    # Set the seed for PyTorch (GPU, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set the seed for rdt HyperTransformer
    ht = HyperTransformer()
    ht.random_state = seed
    print(f"Seed initialized to: {seed}")




def get_latent_representation(input_df, synthesizer, ht):
    # returns the latent space representation of training instances
    initialize_seed_from_env()
    transformed_data = ht.transform(input_df)
    latent_representation = synthesizer.encode_to_latent(transformed_data)
    return latent_representation



def decode_latent_representation(latent_representation, synthesizer, ht):
    # returns the reconstructed samples 
    initialize_seed_from_env()
    decoded_data = synthesizer.decode_from_latent(latent_representation)
    reconstructed_data = ht.reverse_transform(pd.DataFrame(decoded_data))
    return reconstructed_data



def sample_multivariate_and_save(synthesizer, ht, num_samples, embedding_dimension, train_file_path):
    """
    Returns the samples from multivariate normal distribution obtained from the latent representation of training instances.
    """
    initialize_seed_from_env()
    
    df_adult_train = pd.read_csv(train_file_path)
    df_adult_train = df_adult_train.iloc[:, :-1]
    
    num_samples = df_adult_train.shape[0]
    
    base_filename = os.path.splitext(os.path.basename(train_file_path))[0]
    output_filename = f"{embedding_dimension}_multivariate_normal_sampled_instances_from_{base_filename}_{num_samples}_{current_timestamp}.csv"
    output_file_name_multivariate_synthetic =  os.path.join(reconstructed_samples_path, output_filename)
    
    df_adult_train_transformed_latent_representation = get_latent_representation(df_adult_train, synthesizer, ht)
    
    if isinstance(df_adult_train_transformed_latent_representation, torch.Tensor):
        df_adult_train_transformed_latent_representation = df_adult_train_transformed_latent_representation.detach().cpu().numpy()
    print(f"\n type of df_adult_train_transformed_latent_representation: {type(df_adult_train_transformed_latent_representation)}, df_adult_train_transformed_latent_representation's shape: {df_adult_train_transformed_latent_representation.shape}")
    
    # Compute the mean and covariance of the training instances
    mean_vector_train = np.mean(df_adult_train_transformed_latent_representation, axis=0)
    print(f"\n type of mean vector: {type(mean_vector_train)}, mean_vector's shape: {mean_vector_train.shape} | mean_vector: {mean_vector_train}")
    
    covariance_matrix_train = np.cov(df_adult_train_transformed_latent_representation, rowvar=False)
    print(f"\n covariance_matrix_cov :{covariance_matrix_train}, covariance_matrix_cov's shape: {covariance_matrix_train.shape}")
    
    # Generate samples from the multivariate normal distribution
    multivariate_samples = np.random.multivariate_normal(mean_vector_train, covariance_matrix_train, size=num_samples)
    df_multivariate_normal_distribution_samples = pd.DataFrame(multivariate_samples, columns=[f'variable_{i}' for i in range(multivariate_samples.shape[1])])
    multivariate_samples_tensor = torch.tensor(multivariate_samples, dtype=torch.float32, device='cpu')
    reconstructed_data = decode_latent_representation(multivariate_samples_tensor, synthesizer, ht)
    
    # Save the reconstructed data to CSV
    print(f"\nReconstructed data from multivariate normal distribution is saved to file: {output_file_name_multivariate_synthetic}")
    reconstructed_data.to_csv(output_file_name_multivariate_synthetic, index=False)



def sample_normal_and_save(synthesizer, ht, num_samples, embedding_dimension):
    """
    Generates synthetic data samples, reverses the transformation using HyperTransformer,
    and saves the reconstructed synthetic data with a specific filename format.
    """
    initialize_seed_from_env()
    
    # Generate the output file name
    output_file_name = f"{embedding_dimension}_normal_sampled_instances_{num_samples}_{current_timestamp}.csv"
    
    output_file_name_normal_synthetic =  os.path.join(reconstructed_samples_path, output_file_name)
    # Generate synthetic data samples
    sampled_synthetic_data = synthesizer.sample(num_rows=num_samples)
    # Reverse transform to the original format
    reconstructed_synthetic_data = ht.reverse_transform(sampled_synthetic_data)
    print(f"\nSynthetic data sampled from latent space:\n{reconstructed_synthetic_data}")
    # Save to CSV with the generated filename
    reconstructed_synthetic_data.to_csv(output_file_name_normal_synthetic, index=False)
    print(f"Synthetic data saved to {output_file_name_normal_synthetic}")