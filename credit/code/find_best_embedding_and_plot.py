import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def initialize_seed_from_env():
    """
    Initialize the seed for all libraries using the seed stored in the GLOBAL_SEED environment variable.
    """
    seed = int(os.getenv("GLOBAL_SEED", 42))  # Default to 42 if not set
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for Pandas
    pd.core.common.random_state(seed)
    print(f"Seed initialized to: {seed}")


def find_best_loss_last_20_epochs(folder_path):
    # Initialize variables to track the best embedding dimension and loss
    initialize_seed_from_env()
    best_embedding_dim = None
    min_avg_loss_last_20_epochs = float('inf')
    embedding_dims = []
    avg_loss_last_20_epochs = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if "_summary_metrics_" in file_name and file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract embedding dimension and average loss for the last 20 epochs
            embedding_dim = df['Embedding_Dim'].iloc[0]
            avg_loss_last_20_epoch = df['Avg_Loss_Last_20_Epochs'].iloc[0]
            
            # Collect data for analysis
            embedding_dims.append(embedding_dim)
            avg_loss_last_20_epochs.append(avg_loss_last_20_epoch)
            
            # Update the best embedding dimension based on minimum average loss
            if avg_loss_last_20_epoch < min_avg_loss_last_20_epochs:
                min_avg_loss_last_20_epochs = avg_loss_last_20_epoch
                best_embedding_dim = embedding_dim

    # Sort data by embedding dimensions
    sorted_indices = np.argsort(embedding_dims)
    embedding_dims = np.array(embedding_dims)[sorted_indices]
    avg_loss_last_20_epochs = np.array(avg_loss_last_20_epochs)[sorted_indices]

    print(f"Best Embedding Dimension is: {best_embedding_dim} which has minimum loss of {min_avg_loss_last_20_epochs} for last 20 epochs ")
    # return best_embedding_dim, min_avg_loss_last_20_epochs, embedding_dims, avg_loss_last_20_epochs



def find_best_embedding_and_plot(folder_path):
    
    """
    Find the embedding dimension corresponding to the minimum Avg_Loss_Last_Epoch,
    and plot Avg_Loss_Last_Epoch vs. Embedding_Dim.
    Args:
        folder_path (str): Path to the folder containing the CSV files.
    Returns:
        tuple: Best embedding dimension and its corresponding Avg_Loss_Last_Epoch.
    """
    initialize_seed_from_env()
     
    best_embedding_dim = None
    min_avg_loss_last_epoch = float('inf')
    embedding_dims = []
    avg_loss_last_epochs = []

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if "_summary_metrics_" in file_name:
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Collect data for plotting
            embedding_dim = df['Embedding_Dim'].iloc[0]
            avg_loss_last_epoch = df['Avg_Loss_Last_Epoch'].iloc[0]
            embedding_dims.append(embedding_dim)
            avg_loss_last_epochs.append(avg_loss_last_epoch)

            # Check if the current file has the minimum Avg_Loss_Last_Epoch
            if avg_loss_last_epoch < min_avg_loss_last_epoch:
                min_avg_loss_last_epoch = avg_loss_last_epoch
                best_embedding_dim = embedding_dim
    
    # Sort data by embedding dimensions
    sorted_indices = np.argsort(embedding_dims)
    embedding_dims = np.array(embedding_dims)[sorted_indices]
    avg_loss_last_epochs = np.array(avg_loss_last_epochs)[sorted_indices]
    
    # Interpolate for smoother transitions between points
    x_new = np.linspace(embedding_dims.min(), embedding_dims.max(), 100)
    y_new = np.interp(x_new, embedding_dims, avg_loss_last_epochs)

    # Plot Avg_Loss_Last_Epoch vs. Embedding_Dim
    plt.figure(figsize=(10, 6))
    plt.plot(x_new, y_new, linestyle='-', color='blue', label='Avg Loss Last Epoch')
    plt.scatter(embedding_dims, avg_loss_last_epochs, color='blue', label='Embedding Points')
    plt.scatter(best_embedding_dim, min_avg_loss_last_epoch, color='red', label=f'Min Loss: {min_avg_loss_last_epoch:.2f}')
    plt.title("Avg Loss Last Epoch vs. Embedding Dimension")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Avg Loss Last Epoch")
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the folder
    plot_file_path = os.path.join(folder_path, "avg_loss_vs_embedding_dim_smoothed.png")
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Plot saved to {plot_file_path}")

    return best_embedding_dim, min_avg_loss_last_epoch
