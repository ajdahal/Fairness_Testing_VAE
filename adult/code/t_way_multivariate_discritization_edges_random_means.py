import os, sys
import re
import numpy as np
import pandas as pd
import argparse
import operator
import torch
import pickle
from datetime import datetime
import itertools
from lime.discretize import EntropyDiscretizer
from collections import defaultdict
from functools import reduce
from testflows.combinatorics import Covering
from sampling_normal_mutivariate_normal import  get_latent_representation,decode_latent_representation
random_state = 42


current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Format: YYYY-MM-DD_HH-MM
t_way_samples_path = "t_way_samples"



def process_and_save_combinations(combinations, synthesizer, ht, base_filename, combination_type):
    # Convert combinations to PyTorch tensor
    combinations_tensor = torch.tensor(combinations, dtype=torch.float32)
    # Decode from latent space using the combinations
    decoded_data = synthesizer.decode_from_latent(combinations_tensor)
    # Reverse transform to the original format
    reconstructed_data = ht.reverse_transform(pd.DataFrame(decoded_data))
    # Define output file path
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    output_filename = f"{base_name}_{combination_type}_{current_timestamp}.csv"
    output_path = os.path.join(t_way_samples_path, output_filename)
    # Save the reconstructed data to CSV file
    reconstructed_data.to_csv(output_path, index=False)
    print(f"Data sampled from the latent space is saved to \n {output_path}")
    
    
    
    
def generate_all_combinations(bin_dict):
    """
    Generates all possible combinations of arrays from different labels in the provided dictionary.
    Parameters:
    bin_dict (dict): Dictionary where keys are labels and values are arrays.
    Returns:
    list: A list of tuples, each containing a combination of arrays from different labels.
    """
    # Extract the arrays corresponding to each label
    arrays = [bin_dict[label] for label in sorted(bin_dict.keys())]
    # print(f"generate_combinations, arrays: {arrays}")
    # Generate all possible combinations using Cartesian product
    combinations = list(itertools.product(*arrays))
    return combinations



def entropy_discretize_dataframe(latent_representation_multivariate, latent_representation, df_labels_label_column):
    """
    Discretizes each column in the data CSV file using entropy-based discretization.
    """
    if isinstance(latent_representation, torch.Tensor):
        # Detach the tensor and convert to a NumPy array before creating a DataFrame
        latent_representation = pd.DataFrame(
            latent_representation.detach().numpy(), 
            columns=[f'column_{i}' for i in range(latent_representation.shape[1])]
        )
    print(f"\n latent representation: {latent_representation[0:2]} | type: {type(latent_representation)}")
    
    if isinstance(latent_representation_multivariate, torch.Tensor):
        # Detach the tensor and convert to a NumPy array before creating a DataFrame
        latent_representation_multivariate = pd.DataFrame(
            latent_representation_multivariate.detach().numpy(), 
            columns=[f'column_{i}' for i in range(latent_representation_multivariate.shape[1])]
        )
    print(f"\n latent_representation_multivariate: {latent_representation_multivariate[0:2]} | type: {type(latent_representation_multivariate)}")
    
    data = latent_representation
    # Extract labels as a numpy array
    labels = df_labels_label_column.values
    
    print(f"\n entropy_discretize_dataframe: type of data: {type(data)}, shape of data is: {data.shape}")
    print(f"\n entropy_discretize_dataframe: type of labels: {type(labels)}, shape of labels is: {labels.shape}")
    
    
    # Initialize dictionaries to store bin edges and means
    bin_edges_dict = {}
    bin_means_dict = {}
    random_points_from_bins_dict = {}
    
    # Iterate over each column in the DataFrame
    for column in data.columns:
        # Extract the column data as a numpy array
        column_data = data[column].values.reshape(-1, 1)
        column_data_multivariate = latent_representation_multivariate[column].values.reshape(-1,1)
        
        # Initialize the EntropyDiscretizer
        discretizer = EntropyDiscretizer(
            data=column_data,
            categorical_features=[],
            feature_names=[column],
            labels=labels,
            random_state=random_state
        )
        
        # Retrieve bin boundaries for the column
        bin_edges = discretizer.bins(column_data, labels)[0]
        
        
        # Discretize the column data
        discretized_column_old = discretizer.discretize(column_data).flatten()
        discretized_column_old = discretized_column_old.astype(int)
        print(f"\n discretized_column_old | discretized_column's old min: {discretized_column_old.min()} |  discretized_column' old max:{discretized_column_old.max()} | from training data: {discretized_column_old[0:10]}")
        
        
        # Discretize the multivariate data manually using bin_edges
        # Convert bin_edges to include implicit boundaries
        extended_bin_edges = np.concatenate(([-np.inf], bin_edges, [np.inf]))
        
        # Use np.digitize to map data into bins
        discretized_column = np.digitize(column_data_multivariate.flatten(), extended_bin_edges, right=False) - 1
        discretized_column = discretized_column.astype(int)
        print(f"\n discretized_column | discretized_column's min: {discretized_column.min()} |  discretized_column's max:{discretized_column.max()}    | from multivariate data: {discretized_column[0:10]}")
        
        
        ### Update bin edges to include min and max of the data ###
        label_min_max_count_dict = defaultdict(lambda: [sys.maxsize, -sys.maxsize, 0])
        for label, value in zip(discretized_column, column_data_multivariate.flatten()):
            label_min_max_count_dict[label][0] = min(label_min_max_count_dict[label][0], value)  # min
            label_min_max_count_dict[label][1] = max(label_min_max_count_dict[label][1], value)  # max
            label_min_max_count_dict[label][2] += 1  # count
        
        
        # Sort label_min_max_count_dict by keys
        sorted_label_min_max_count_dict = {
            label: (min_val, max_val, count)
            for label, (min_val, max_val, count) in sorted(label_min_max_count_dict.items(), key=lambda x: x[0])
        }
        
        bin_edges_dict[column] = bin_edges
        
        #for x in discretized_column:
        #    print(f"discretized_column's x: {x} | type is: {type(x)}")
        # print(f"discretized_column: {discretized_column}")
        
        # Initialize list for means and random points
        bin_means = []
        random_points = []
        
        # Calculate means and random points for each label
        for label, (min_val, max_val, count) in sorted_label_min_max_count_dict.items():

            #print(f"label: {label} | type of label: {type(label)}")
            label_indices = np.where(discretized_column == label)[0]      # finds  the indices where label matches the current bin
            label_points = column_data_multivariate[label_indices].flatten()            # Ensure it's 1-dimensional
            
            # If you have to calculate median or medeoid, this is the place to start ---> 
            # Calculate mean
            mean = label_points.mean() if len(label_points) > 0 else np.nan
            bin_means.append(mean)
            
            # Select a random point
            if len(label_points) > 0:
                random_point = np.random.choice(label_points)
            else:
                random_point = None  # Handle empty bins
            random_points.append(random_point)
        
        # Store results
        bin_means_dict[column] = bin_means
        random_points_from_bins_dict[column] = random_points
        
        ### Print results ###
        print(f"column: {column}")
        print(f"bin_edges: {bin_edges}")
        print(f"bin_means: {bin_means}")
        print(f"Random points from bins: {random_points}")
        print(f"Sorted Label Min-Max-Count Dictionary: {sorted_label_min_max_count_dict} \n")
    return bin_edges_dict, bin_means_dict, random_points_from_bins_dict


def t_way_multivariate_combinations(synthesizer, ht, embedding_dim, multivariate_filename, filename_latent_representation, filename_labels, label_column_name):
        
        max_t = 5                                              # Change this hyperparameter - number of t_way combinations to form as necessary
        print(f"\n multivariate_filename: {multivariate_filename}, \n filename_latent_representation: {filename_latent_representation}")
        
        df_data = pd.read_csv(filename_latent_representation)
        df_without_class = df_data.drop('income', axis=1) if 'income' in df_data.columns else df_data
        
        df_labels = pd.read_csv(filename_labels)
        df_labels_label_column = df_labels[label_column_name]

        latent_representation = get_latent_representation(df_without_class, synthesizer, ht)
        
        df_multivariate = pd.read_csv(multivariate_filename)
        df_multivariate_without_class = df_multivariate.drop('income', axis=1) if 'income' in df_multivariate.columns else df_multivariate
        latent_representation_multivariate = get_latent_representation(df_multivariate_without_class, synthesizer, ht)
        
        
        bin_edges, bin_means, random_points_from_bins  = entropy_discretize_dataframe(latent_representation_multivariate, latent_representation, df_labels_label_column)
        print(f"\n\n   bin_edges: {bin_edges}")
        print(f"\n\n   bin_means: {bin_means}")
        print(f"\n\n   random_points_from_bins_dict: {random_points_from_bins}")
        
        # Calculate the number of combinations for bin edges
        element_counts_bin_edges = {label: len(values) for label, values in bin_edges.items()}
        product_of_counts_bin_edges = reduce(operator.mul, element_counts_bin_edges.values(), 1)
        print(f"The product of the number of elements for bin_edges is: {product_of_counts_bin_edges}")
        # combinations_bin_edges = generate_all_combinations(bin_edges)
        # print(f"combinations_bin_edges type: {type(combinations_bin_edges)} length is: {len(combinations_bin_edges)}")
        # print(f"some combinations are: {combinations_bin_edges[0:5]}")


        # Calculate the number of combinations for bin means
        # element_counts_bin_means = {label: len(values) for label, values in bin_means.items()}
        # product_of_counts_bin_means = reduce(operator.mul, element_counts_bin_means.values(), 1)
        # print(f"The product of the number of elements for bin_means is: {product_of_counts_bin_means}")
        # combinations_bin_means = generate_all_combinations(bin_means)
        # print(f"combinations_bin_means type: {type(combinations_bin_means)}, length is: {len(combinations_bin_means)}")
        
        
        # Calculate the number of combinations for random points from the bin
        # combinations_bin_random_points = generate_all_combinations(random_points_from_bins)
        # print(f"combinations_bin_random_points type: {type(combinations_bin_random_points)}, length is: {len(combinations_bin_random_points)}")
        
        bin_edges_list = {key: value.tolist() for key, value in bin_edges.items()}
        
        
        #print(f"bin_edges_list: {bin_edges_list}")
        #print(f"bin_edges_list: {bin_edges_list}")
        #print(f"bin_means_list: {bin_means_list}")
        
        for t_way in range(2, max_t):
            combination_type = str(t_way) + "_way_covering_array" + "_bin_edges"
            covering_array = Covering(bin_edges_list, strength=t_way)
            t_way_combinations = [tuple(d.values()) for d in covering_array ]
            process_and_save_combinations(t_way_combinations,synthesizer, ht, base_filename=multivariate_filename, combination_type=combination_type)
        

        for t_way in range(2, max_t):
            combination_type = str(t_way) + "_way_covering_array" + "_bin_means"
            covering_array = Covering(bin_means, strength=t_way)
            t_way_combinations = [tuple(d.values()) for d in covering_array ]
            process_and_save_combinations(t_way_combinations,synthesizer, ht, base_filename=multivariate_filename, combination_type=combination_type)
        
        
        for t_way in range(2, max_t):
            combination_type = str(t_way) + "_way_covering_array" + "_bin_random_points"
            covering_array = Covering(random_points_from_bins, strength=t_way)
            t_way_combinations = [tuple(d.values()) for d in covering_array ]
            process_and_save_combinations(t_way_combinations,synthesizer, ht, base_filename=multivariate_filename, combination_type=combination_type)