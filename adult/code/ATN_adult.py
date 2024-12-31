# import necessary libraries

import random
import pandas as pd
import numpy as np
from sdmetrics.single_table import KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity
import os, argparse
from datetime import datetime
from rdt import HyperTransformer
import csv


current_timestamp = datetime.now().strftime("%Y-%m-%d_%H")


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
    # Set the seed for rdt HyperTransformer
    ht = HyperTransformer()
    ht.random_state = seed
    print(f"Seed initialized to: {seed}")



def load_and_equalize_data(test_data_file, adv_data_file, mode):
    
    df_xs = pd.read_csv(test_data_file)
    df_xt = pd.read_csv(adv_data_file)
    
    # Replacing underscores with hyphens in column headers
    df_xs.columns = [col.replace('_', '-') for col in df_xs.columns]
    df_xt.columns = [col.replace('_', '-') for col in df_xt.columns]
    
    if mode == "equal":
        # Determine which DataFrame has fewer rows
        if len(df_xs) > len(df_xt):
            # Sample the larger DataFrame to match the smaller one
            df_xs = df_xs.sample(n=len(df_xt), random_state=42)
        elif len(df_xt) > len(df_xs):
            # Sample the larger DataFrame to match the smaller one
            df_xt = df_xt.sample(n=len(df_xs), random_state=42)
    # If "not-equal" is specified, the dataframes remain unchanged
    return df_xs, df_xt

# Function to convert boolean/numeric columns to categorical
def convert_boolean_to_categorical(df, boolean_cols):
    for col in boolean_cols:
        if df[col].nunique() == 2:  # Assuming a binary column (0 and 1)
            df[col] = df[col].astype('category')
    return df


def get_atn(test_data_file, adv_data_file, mode):
    
    # Load and equalize data if necessary
    df_xs, df_xt = load_and_equalize_data(test_data_file, adv_data_file, mode)
    print(f"After load_and_equalize_data function is run: df_xs shape: {df_xs.shape}, df_xt shape: {df_xt.shape}")
    
    # continuous_cols = ['age', 'hours-per-week']
    # categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'race', 'sex']   # sex is removed
    
    continuous_cols = ['age','education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    
    # Call function to convert boolean columns (like 'income') to categorical
    # df_xs = convert_boolean_to_categorical(df_xs, ['income'])
    # df_xt = convert_boolean_to_categorical(df_xt, ['income'])
    
    # Calculate KSTest for continuous columns
    ks_scores= []
    for col in continuous_cols:
        ks_score = KSComplement.compute(real_data = df_xs[[col]], synthetic_data = df_xt[[col]])
        ks_scores.append(ks_score)


    # Calculate Total Variation Distance Complement for categorical columns
    TV_scores = []
    for col in categorical_cols:
        tv_score = TVComplement.compute(real_data = df_xs[[col]], synthetic_data = df_xt[[col]])
        TV_scores.append(tv_score)


    # Calculate Correlation Similarity between different columns
    corr_scores = []
    for i in range(len(continuous_cols)):
        for j in range(i + 1, len(continuous_cols)):
            col1 = continuous_cols[i]
            col2 = continuous_cols[j]
            
            # Does either column has zero variance ( i.e. the column has constant terms in it)
            if df_xs[col1].std() == 0 or df_xt[col2].std() == 0 or df_xs[col1].std() == 0 or df_xt[col2].std() == 0:
                print(f"Skipping correlation calculation for columns {col1} and {col2} due to zero variance.")
                continue
            try:
                # Compute correlation similarity
                corr_score = CorrelationSimilarity.compute(real_data=df_xs[[col1, col2]], synthetic_data=df_xt[[col1, col2]], coefficient='Pearson')
                corr_scores.append(corr_score)
            except Exception as e:
                print(f"Error computing correlation for columns {col1} and {col2}: {e}")
                
                
    # Calculate Contingency Similarity between different columns
    contingency_scores = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            col1 = categorical_cols[i]
            col2 = categorical_cols[j]
            
            # Check if either column has zero variance
            if df_xs[col1].nunique() == 1 or df_xt[col1].nunique() == 1 or df_xs[col2].nunique() == 1 or df_xt[col2].nunique() == 1:
                print(f"Skipping contingency score calculation for columns {col1} and {col2} due to zero variance.")
                continue
            try:
                # Compute contingency similarity
                contingency_score = ContingencySimilarity.compute(real_data=df_xs[[col1, col2]], synthetic_data=df_xt[[col1, col2]])
                contingency_scores.append(contingency_score)
            except Exception as e:
                print(f"Error computing contingency value for columns {col1} and {col2}: {e}")
                
    # print(f"ks_scores: {[round(x,2) for x in ks_scores]},\nTV_scores: {[round(x,2) for x in TV_scores]},\ncorr_scores: {[round(x,2) for x in corr_scores]},\ncontingency scores: {[round(x,2) for x in contingency_scores]}")
    all_scores = np.mean(ks_scores + TV_scores) + np.mean(corr_scores + contingency_scores)
    atn_score = 0.5 * (np.round(all_scores, decimals=2))
    return atn_score
    print(f"Average Tabular Naturalness (ATN) score for files {test_data_file} with shape {df_xs.shape} and {adv_data_file} with shape {df_xt.shape}: {atn_score}")



def process_and_save_atn_file_results(results_dir, test_data_file, adv_data_file, mode):
    """
    Process input files to compute ATN score and save results to a CSV file in a specified directory.
    """
    # Compute ATN score
    atn_score = get_atn(test_data_file, adv_data_file, mode)

    # Read the number of rows in each file using pandas
    try:
        num_rows_test = pd.read_csv(test_data_file).shape[0]
        num_rows_adv = pd.read_csv(adv_data_file).shape[0]
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Ensure results_dir exists
    os.makedirs(results_dir, exist_ok=True)

    # Create the output file name with a timestamp
    current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file = os.path.join(results_dir, f"ATN_results_two_way_{os.path.splitext(os.path.basename(test_data_file))[0]}_{current_timestamp}.csv")

    # Prepare data for CSV
    row_data = {
        "First_Filename": test_data_file,
        "Number_of_Rows_First_File": num_rows_test,
        "Second_Filename": adv_data_file,
        "Number_of_Rows_Second_Filename": num_rows_adv,
        "ATN_Score": atn_score,
        "Mode": mode
    }

    # Write to the CSV file
    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = [
            "First_Filename", 
            "Number_of_Rows_First_File", 
            "Second_Filename", 
            "Number_of_Rows_Second_Filename", 
            "ATN_Score", 
            "Mode"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Write header and row data
        writer.writeheader()
        writer.writerow(row_data)
    print(f"Results saved to {output_file}")




def calculate_and_save_atn_scores(train_file_path, t_way_samples_dir, results_dir, mode='equal'):
    """
    Calculate ATN scores for all files in the t_way_samples_dir with respect to train_file_path,
    and save the results to a CSV file, including file shapes.

    Args:
        train_file_path (str): Path to the training file.
        t_way_samples_dir (str): Directory containing test files.
        results_dir (str): Directory to save the results CSV file.
        mode (str): Mode to use in ATN calculation (default is 'equal').
    """
    
    initialize_seed_from_env()
    
    # Get the train file name (without path)
    train_file_name = os.path.basename(train_file_path)

    # Load and get the shape of the training file
    train_df = pd.read_csv(train_file_path)
    train_file_shape = train_df.shape[0]

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, f"ATN_results_{current_timestamp}.csv")

    # Initialize results list
    results = []

    # Iterate through all files in t_way_samples_dir
    for test_file_name in os.listdir(t_way_samples_dir):
        test_file_path = os.path.join(t_way_samples_dir, test_file_name)

        # Skip files that are not csv
        if not test_file_name.endswith(".csv"):
            continue
        
        # Load and get the shape of the test file
        test_df = pd.read_csv(test_file_path)
        test_file_shape = test_df.shape[0]

        # Calculate ATN score
        ATN = get_atn(train_file_path, test_file_path, mode)

        # Append the result
        results.append({
            "Train_File_Name": train_file_name,
            "Number_of_Rows_Train_File": train_file_shape,
            "Test_File_Name": test_file_name,
            "Number_of_Rows_Test_File": test_file_shape,
            "ATN_Score": ATN,
            "Mode": mode
        })
        
        results = sorted(results, key=lambda x: x["Train_File_Name"])
        
        # Print the result
        print(f"Train File: {train_file_name} (#Rows: {train_file_shape}), "
              f"Test File: {test_file_name} (#Rows: {test_file_shape}), "
              f"ATN Score: {ATN}, Mode: {mode}")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(results_file_path, index=False)
    print(f"\nResults saved to {results_file_path}")
