import os
import sys
import random
import pickle
import torch
import glob
import subprocess
import numpy as np
import pandas as pd
from rdt import HyperTransformer
from sdv.single_table import TVAESynthesizer

# Add the 'code' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

import train_models_credit
from download_split_dataset import process_and_split_data
from preprocess_dataset import preprocess_data
from train_vae import train_vae
from find_best_embedding_and_plot import find_best_embedding_and_plot, find_best_loss_last_20_epochs
from sampling_normal_mutivariate_normal import sample_normal_and_save, sample_multivariate_and_save, get_latent_representation
from t_way_discritization_edges_random_means import t_way_combinations
from ATN_credit import calculate_and_save_atn_scores, process_and_save_atn_file_results
from t_way_multivariate_discritization_edges_random_means import t_way_multivariate_combinations
from discriminatory_sex_age_combinations_credit import process_models, fix_header_csv_file



# Define file paths and directories
base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
t_way_samples_dir = os.path.join(base_dir, "t_way_samples")
sampling_results_dir = os.path.join(base_dir, "sampling_results")
results_dir = os.path.join(base_dir, "results")
output_dir = os.path.join(base_dir, "output")


train_file_path = os.path.join(dataset_dir, "credit_train.csv")
test_file_path = os.path.join(dataset_dir, "credit_test.csv")


def set_global_seed(seed):
    """
    Set the global seed for all libraries and store it as an environment variable.
    """
    # Save the seed to an environment variable for global access
    os.environ["GLOBAL_SEED"] = str(seed)
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
    # Ensure deterministic behavior in PyTorch (if required)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    # Set the seed for rdt HyperTransformer
    ht = HyperTransformer()
    ht.random_state = seed
    print(f"Global seed set to: {seed}")




def main():
    
    # Step 1: Check if the dataset files exist
    if not (os.path.exists(train_file_path) and os.path.exists(test_file_path)):
        print("Downloading and splitting the dataset...")
        train_data, test_data = process_and_split_data(dataset_dir)
        print("Dataset downloaded and split successfully.")
    else:
        print("Dataset files already exist. Loading the files...")
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
    
    # Now you can use train_data and test_data
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Step 2: Preprocess the training dataset
    print("Preprocessing the training dataset...")
    X_train_transformed, X_test_transformed = preprocess_data(models_dir, train_file_path=train_file_path, test_file_path=test_file_path)
    print("Training dataset preprocessed successfully.")
    
    print("Transformed Training Data (Sample):")
    print(X_train_transformed.head(), (X_train_transformed.shape)[1])
    
    # Step 3: Train VAE for each embedding dimension
    half_the_columns = (train_data.shape[1] - 1) // 2
    print(f"\nNumber of columns in training data from excluding class column: {half_the_columns}")
    embedding_dim_min = max(2, half_the_columns - 4)  # At least 2 as the minimum embedding dimension
    embedding_dim_max = half_the_columns + 4

    embedding_dims = list(range(embedding_dim_min, embedding_dim_max + 1))
    print(f"Embedding dimensions: {embedding_dims}")
    
    for embedding_dim in embedding_dims:
        print(f"Training VAE with embedding dimension: {embedding_dim}")
        train_vae(X_train_transformed=X_train_transformed, embedding_dim=embedding_dim, output_dir=models_dir)
    
    # Step 4: Find the embedding dimension corresponding to the minimum reconstruction loss 
    best_embedding_dim, min_avg_loss_last_epoch = find_best_embedding_and_plot(models_dir)
    find_best_loss_last_20_epochs(models_dir)
    print(f"Best Embedding Dimension: {best_embedding_dim}, Minimum Avg_Loss_Last_Epoch: {min_avg_loss_last_epoch}")
    
    embedding_dimension = best_embedding_dim
    # Step 4.5 (optional): Find samples from normal distribution and multivariate normal distribution using this information
    # Load the synthesizer model
    model_path = next((os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith(f"{embedding_dimension}_tvae_credit_synthesizer_model") and f.endswith(".pkl")), None)
    
    synthesizer = TVAESynthesizer.load(model_path)
    print(f"\n VAE Model loaded successfully : {model_path}")
    
    # Load the HyperTransformer
    hypertransformer_file_path = next((os.path.join(models_dir, f) for f in os.listdir(models_dir) if "tvae_hypertransformer.pkl" in f), None)
    with open(hypertransformer_file_path, 'rb') as f:
        ht = pickle.load(f)
    print(f"\n HyperTransformer loaded successfully: {hypertransformer_file_path}")
    
    num_samples = train_data.shape[0]
    
    sample_normal_and_save(synthesizer, ht, num_samples, embedding_dimension)
    sample_multivariate_and_save(synthesizer, ht, num_samples, embedding_dimension, train_file_path)
    
    # Step 5: find t_way_representation
    file_to_find_tway = train_file_path
    filename_labels = train_file_path
    label_column_name = "class"
    t_way_combinations(synthesizer, ht, embedding_dimension, file_to_find_tway, filename_labels, label_column_name)
    
    
    multivariate_file_pattern =  os.path.join(sampling_results_dir, "*multivariate_normal_sampled_instances*.csv")
    matching_files = glob.glob(multivariate_file_pattern)
    multivariate_filename = matching_files[0] if matching_files else None
    t_way_multivariate_combinations(synthesizer, ht, embedding_dimension, multivariate_filename, train_file_path, filename_labels, label_column_name)
    
    # Step 6: find ATN for t_way_samples with respect to training data
    calculate_and_save_atn_scores(train_file_path, t_way_samples_dir, results_dir, mode='not-equal')  # mode: 'equal' or 'not-equal'    
    
    # Step 7: Find the ratio of discriminatory instances
    # Train the machine learning models
    train_models_credit.main(train_file_path, test_file_path, models_dir)
    
    t_way_patel = fix_header_csv_file("../Patel_Data/tWay_Concrete_TC/GermanCredit_AI360_Modified_2way_concrete_TC_with_constraint.csv")
    t_way_ours = glob.glob("t_way_samples/credit_train_2_way_covering_array_bin_means*.csv")
    
    dataset_patel = fix_header_csv_file("../Patel_Data/Dataset/GermanCredit_AI360_Modified.csv")
    dataset_ours = os.path.join(dataset_dir, "credit_all_data.csv")
    
    # Find the discriminatory instances for generated t-way instances
    process_models(t_way_patel, models_dir, output_dir, results_dir)

    print(f"\n t_way_ours: {t_way_ours}")
    for t_way_file in t_way_ours:
        process_models(t_way_file, models_dir, output_dir, results_dir)
    
    # Step 8: Find ATN
    for t_way_file in t_way_ours:
        process_and_save_atn_file_results(results_dir, test_data_file = t_way_file, adv_data_file=dataset_ours, mode='not-equal')
    process_and_save_atn_file_results(results_dir, test_data_file = t_way_patel, adv_data_file=dataset_patel, mode='not-equal')



if __name__ == "__main__":
    SEED = 42
    set_global_seed(SEED)
    main()