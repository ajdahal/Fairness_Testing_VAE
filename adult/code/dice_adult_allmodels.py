# Necessary Imports
import os
import re
import pickle
import random
import json
import glob
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
from collections import defaultdict
import tensorflow as tf

# import Dice
import dice_ml


current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")   # Format: YYYY-MM-DD_HH-MM

base_dir = os.path.dirname("../")
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
dice_results_dir = os.path.join(base_dir,"DICE_results")
t_way_samples_dir = os.path.join(base_dir,"t_way_samples")
train_file_path = os.path.join(dataset_dir, "adult_train.csv")
test_file_path = os.path.join(dataset_dir, "adult_test.csv")



def set_global_seed(seed=42):
    """
    Set the global seed for reproducibility across Python, NumPy, and TensorFlow.
    Args:
        seed (int): The seed value to set.
    """
    
    # Python's random module
    random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    # Optional: Ensure GPU determinism (for TensorFlow)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global seed set to {seed}")


# Set the seed for reproducibility
set_global_seed(42)



# Load Dataset
def load_dataset(filepath):
    dataset = pd.read_csv(filepath, index_col=None)
    target = dataset['income']
    datasetX = dataset.drop('income', axis=1)
    return dataset, datasetX, target


# Load Model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model



def get_model_paths(models_dir):
    """
    Returns:
    list: List of full model paths with the latest consistent date section for specified models.
    """
    # Allowed model names
    allowed_models = {"logistic_regression", "nn", "random_forest", "svm"}
    
    # Initialize a defaultdict to group files by date
    models_by_date = defaultdict(list)

    # Regular expression to capture valid filenames and the date section
    pattern = re.compile(r"^adult__({})__(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}})".format('|'.join(allowed_models)))

    # List all files in the models directory
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            model_name = match.group(1)  # Extract the model name
            date_section = match.group(2)  # Extract the date section
            full_path = os.path.join(models_dir, filename)
            models_by_date[date_section].append(full_path)

    # Select the files with the most recent date section
    if models_by_date:
        latest_date_section = max(models_by_date.keys())  # Get the latest date section
        model_paths = models_by_date[latest_date_section]
    else:
        model_paths = []  # Handle case where no files match the pattern
        
    return model_paths





def save_counterfactuals(json_list, model_used, query_file_name_without_extension):
    
    # Define the columns for the dataset
    columns = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    
    # Initialize the results list
    changes_list = []
    # Iterate through the data and counterfactuals
    for idx, cfs in enumerate(json_list["cfs_list"]):
        if cfs is not None:
            test_instance = json_list["test_data"][idx][0]  # Original data instance
            test_instance_dict = dict(zip(columns, test_instance))  # Convert to dictionary for easy comparison
            
            # print(f"\n test_instance_dict: {test_instance_dict}")
            # Add original attribute index
            index_original_attribute = idx + 1  # Position from the start of the test_instance list
            
            # Process counterfactuals
            for cf in (cfs if isinstance(cfs[0], list) else [cfs]):  # Handle single or multiple counterfactuals
                cf_dict = dict(zip(columns, cf))  # Counterfactual instance as dictionary
                
                # Find changed attributes excluding income
                changed_attributes = [
                        column for column in columns 
                        if test_instance_dict[column] != cf_dict[column] and test_instance_dict['income'] != cf_dict['income']
                ]
                
                # Prepare the row for output
                # Prepare the row for output without including 'income'
                original_values = ", ".join(
                    f"{test_instance_dict[col]}" for col in changed_attributes) # if col != 'income') --> include income

                counterfactual_values = ", ".join(
                    f"{cf_dict[col]}" for col in changed_attributes) # if col != 'income') --> include income
                
                
                changes_list.append({
                    "Index_Original_Attribute": index_original_attribute,
                    "Original Attributes": original_values,
                    "Counterfactual Attributes": counterfactual_values,
                    "Changed Attributes": ", ".join(changed_attributes)
                })
                
                
            # Add a gap row
            changes_list.append({
                "Index_Original_Attribute": "",
                "Original Attributes": "",
                "Counterfactual Attributes": "",
                "Changed Attributes": ""
            })


    # Create a DataFrame for better visualization
    changes_df = pd.DataFrame(changes_list)
    
    # Display the result
    # print(changes_df)

    # Ensure the column 'Index_Original_Attribute' exists and is not empty
    if 'Index_Original_Attribute' in changes_df.columns:
        # Filter out empty or non-integer values
        valid_indices = changes_df['Index_Original_Attribute'].dropna().astype(str)
        valid_indices = valid_indices[valid_indices.str.isdigit()]  # Keep only strings that represent digits
        unique_count = valid_indices.nunique()
        total_count = len(valid_indices)
        print(f"Total number of instances for which counterfactuals are generated: {unique_count}")
        print(f"Total number of counterfactuals: {total_count}")
    
    output_file_name = "Counterfactuals_" + str(query_file_name_without_extension) + "_" + str(model_used) + "_" + ".csv"
    
    # Join the directory with the output file name
    output_file_path = os.path.join(dice_results_dir, output_file_name)
    
    print(f"Counterfactuals saved to the file: {output_file_path}")
    changes_df.to_csv(output_file_path, index=False)
    # return unique_count



def print_counterfactuals(json_list):
    # Initialize a list to hold the combined data and counterfactuals
    combined_data = []
    # Process `cfs_list` and combine corresponding elements from `test_data`
    for idx, cfs in enumerate(json_list["cfs_list"]):
        if cfs is not None:
            test_data_instance = json_list["test_data"][idx][0]  # Get the corresponding test_data instance
            # Combine the data and all its counterfactuals in a single row
            combined_data.append({
                "data": test_data_instance,
                "cfs": cfs if isinstance(cfs[0], list) else [cfs]  # Handle list or list of lists
            })
    print("\n\n")
    # Display the combined result
    
    for row in combined_data:
        print(row)



# Main Function
def main():
    # Load the dataset
    adult_train_df, x_train, y_train = load_dataset(train_file_path)
    
    continuous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # Define numerical and categorical features
    categorical_features = x_train.columns.difference(continuous_features)
    
    query_file_name = next(iter(glob.glob(os.path.join(t_way_samples_dir, "adult_train_2_way_covering_array_bin_means*.csv"))), None) 

    query_file_name_without_extension = os.path.splitext(os.path.basename(query_file_name))[0]
    
    # Configurable hyperparameters
    t_way_instances = pd.read_csv(query_file_name)             # change this
    t_way_instances_without_income = t_way_instances.drop('income', axis=1) if 'income' in t_way_instances.columns else t_way_instances
    
    print(f"\n File used for query is: {query_file_name}")
    
    query_instances = t_way_instances_without_income            # change this
    features_to_vary = ['sex', 'race']
    total_CFs = 1
    backend = "sklearn"
    outcome_name = 'income'
    
    print(f"Number of query_rows are: {query_instances.shape[0]}")
    
    # Initialize DiCE data object
    d = dice_ml.Data(dataframe=adult_train_df, continuous_features=continuous_features, outcome_name=outcome_name)
    
    # Models to test
    model_paths = get_model_paths(models_dir)
    print(f"\n model paths are: {model_paths}")
    
    
    # Generate counterfactuals for each model
    for model_path in model_paths:
        print(f"Processing model: {model_path}")
        model = load_model(model_path)
        
        # Initialize DiCE model object
        m = dice_ml.Model(model=model, backend=backend)

        # Initialize the counterfactual generation method
        exp_random = dice_ml.Dice(d, m, method="random")
        
        possible_counterfactuals = exp_random.generate_counterfactuals(
                                        query_instances,
                                        total_CFs=total_CFs, 
                                        features_to_vary=features_to_vary, 
                                        desired_class="opposite", 
                                        verbose=False)
        
        json_list = possible_counterfactuals.to_json()
        print(f"possible_counterfactuals: {json_list}")
        
        # visualize the results
        # possible_counterfactuals.visualize_as_dataframe(show_only_changes=True)
        
        
        if isinstance(json_list, str):
            json_list = json.loads(json_list)
        
        print_counterfactuals(json_list)
        save_counterfactuals(json_list, os.path.splitext(os.path.basename(model_path))[0], query_file_name_without_extension)
    
    
    # For Neural Network
    
    # Identify the model file path
    nn_model_path = next((os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith("adult__nn_") and f.endswith(".h5")), None)
    # print(f"\n nn_model_path: {nn_model_path}")

    # Check if the file path is valid
    if nn_model_path and os.path.exists(nn_model_path):
        nn_model = tf.keras.models.load_model(nn_model_path)
        print("Model loaded successfully.")
    else:
        print("No matching .h5 model file found.")
    
    backend = 'TF'+tf.__version__[0]
    print(f"\n the backend used is: {backend}")
    
    # Initialize DiCE model object
    m_nn = dice_ml.Model(model=nn_model, backend=backend, func="ohe-min-max") #, func="ohe-min-max"

    # Initialize the counterfactual generation method
    exp = dice_ml.Dice(d, m_nn, method="random")
    possible_counterfactuals = exp.generate_counterfactuals(query_instances,total_CFs=total_CFs,  features_to_vary=features_to_vary, desired_class="opposite",verbose=False)
    
    json_list = possible_counterfactuals.to_json()
    print(f"possible_counterfactuals: {json_list}")

    if isinstance(json_list, str):
        json_list = json.loads(json_list)
    
    print_counterfactuals(json_list)
    save_counterfactuals(json_list, os.path.splitext(os.path.basename(nn_model_path))[0], query_file_name_without_extension)
    
    
    
# Entry Point
if __name__ == "__main__":
    main()