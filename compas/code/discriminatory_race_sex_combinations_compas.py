import os
import pickle
import random
import re
import joblib
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import numpy as np


base_dir = os.path.dirname("./")
output_dir = os.path.join(base_dir, "output")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")


def set_global_seed(seed=42):
    """
    Set the global seed for reproducibility across Python, NumPy, and TensorFlow.
    Args:
        seed (int): The seed value to set.
    """
    
    # Python's random module
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # TensorFlow
    tf.random.set_seed(seed)
    # GPU determinism (for TensorFlow)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # If using TF >= 2.8
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    print(f"Global seed set to {seed}")


# Set the seed for reproducibility
set_global_seed(42)



def load_preprocessor(export_file_name="compas_preprocessor.pkl"):
    """
    Load a preprocessor from a file.
    """
    import_path = os.path.join(models_dir, export_file_name)
    preprocessor = joblib.load(import_path)
    print(f"Preprocessor loaded from {import_path}")
    return preprocessor



def append_results_to_csv(results, results_dir, input_csv_filename, output_filename_suffix="_discriminatory_instances_results.csv"):
    """
    Append results to a consolidated CSV file in the output directory, dynamically naming it based on input CSV filename.
    """
    
    # Derive the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(input_csv_filename))[0]
    
    # Create the output filename dynamically
    output_filename = f"{base_name}{output_filename_suffix}"
    output_file_path = os.path.join(results_dir, output_filename)

    # Convert results to a DataFrame
    results_df = pd.DataFrame([results])

    # Check if the file exists and append or create a new file
    if os.path.exists(output_file_path):
        # Append to existing file
        existing_df = pd.read_csv(output_file_path)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    else:
        # Create a new file
        combined_df = results_df
    
    # Save the combined DataFrame to the file
    combined_df.to_csv(output_file_path, index=False)
    print(f"Results appended to {output_file_path}")



def get_model_paths(models_dir):
    """
    Get the paths of valid model files in the directory.
    """
    # Allowed model names
    allowed_models = {"logistic_regression", "nn", "random_forest", "svm"}
    
    # List to collect valid model paths
    model_paths = []
    
    # Regular expression to capture valid filenames
    pattern = re.compile(r"^compas__({})__\.(pkl|h5)$".format('|'.join(allowed_models)))

    # List all files in the models directory
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            # Extract model name and extension (optional for debugging/logging)
            model_name, extension = match.groups()
            full_path = os.path.join(models_dir, filename)
            model_paths.append(full_path)

    # Sort paths alphabetically (optional)
    model_paths = sorted(model_paths)
    
    print(f"\nValid model paths: {model_paths}")
    return model_paths


def generate_combinations_from_csv(df, privileged_gender="Male", privileged_race="Caucasian"):
    """
    Generate original and alternative combinations for each row in a dataframe.
    """
    non_privileged_genders = ['Female']
    non_privileged_races = ['African-American', 'Hispanic', 'Other', 'Asian', 'Native American']
    
    print(f"\n non_privileged_genders: {non_privileged_genders}")
    print(f"\n non_privileged_races: {non_privileged_races}")

    combined_data = []
    for _, row in df.iterrows():
        original_instance = row.to_dict()
        combined_data.append(original_instance)

        current_gender = row['sex']
        current_race = row['race']
        
        if current_gender == privileged_gender and current_race == privileged_race:
            combined_data.append({**original_instance, 'sex': random.choice(non_privileged_genders), 'race': privileged_race})
            combined_data.append({**original_instance, 'sex': random.choice(non_privileged_genders), 'race': random.choice(non_privileged_races)})
            combined_data.append({**original_instance, 'sex': privileged_gender, 'race': random.choice(non_privileged_races)})
        elif current_gender in non_privileged_genders and current_race == privileged_race:
            combined_data.append({**original_instance, 'sex': privileged_gender, 'race': privileged_race})
            combined_data.append({**original_instance, 'sex': random.choice(non_privileged_genders), 'race': random.choice(non_privileged_races)})
            combined_data.append({**original_instance, 'sex': privileged_gender, 'race': random.choice(non_privileged_races)})
        elif current_gender in non_privileged_genders and current_race in non_privileged_races:
            combined_data.append({**original_instance, 'sex': privileged_gender, 'race': random.choice(non_privileged_races)})
            combined_data.append({**original_instance, 'sex': random.choice(non_privileged_genders), 'race': privileged_race})
            combined_data.append({**original_instance, 'sex': privileged_gender, 'race': privileged_race})
        elif current_gender == privileged_gender and current_race in non_privileged_races:
            combined_data.append({**original_instance, 'sex': random.choice(non_privileged_genders), 'race': privileged_race})
            combined_data.append({**original_instance, 'sex': random.choice(non_privileged_genders), 'race': random.choice(non_privileged_races)})
            combined_data.append({**original_instance, 'sex': privileged_gender, 'race': privileged_race})
    return combined_data



def find_discriminatory_instances(input_file, model_path, output_file, preprocessor=None, is_neural_network=False):
    
    """
    Identify discriminatory instances by generating predictions for all combinations
    and comparing with the original instance. Handles both traditional ML models
    and neural networks.
    Returns: Results dictionary containing details of discriminatory analysis.
    """
    
    df = pd.read_csv(input_file).drop(columns=['two_year_recid'], errors='ignore')
    df.columns = ['sex','age','age_cat','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','c_charge_degree']

    if is_neural_network:
        # Load neural network model
        model = tf.keras.models.load_model(model_path)
    else:
        # Load ML model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

    combined_data = generate_combinations_from_csv(df)
    
    # Convert combined_data to a DataFrame
    combined_data_df = pd.DataFrame(combined_data)

    results = []
    discriminatory_instances_count = 0
    first_discrepancy_index = None
    output_data = []

    current_original_index = 0

    for i in range(0, len(combined_data_df), 4):        # Each set contains 1 original + 3 alternatives
        original_instance = combined_data_df.iloc[i]
        alternatives = combined_data_df.iloc[i + 1:i + 4]
        
        # print(f"\n original_instance: {original_instance}")
        # print(f"\n alternatives: {alternatives}")
        
        current_original_index += 1  # Increment index for each new set

        if is_neural_network:
            # Preprocess and predict for neural network
            original_transformed = preprocessor.transform(pd.DataFrame([original_instance]))
            original_prediction = (model.predict(original_transformed) > 0.5).astype(int).flatten()[0]
            # print(f"\n original_prediction: {original_prediction}")
            
            alternatives_transformed = preprocessor.transform(alternatives)
            # print(f"\n alternatives_transformed: {alternatives_transformed}")
            
            alternative_predictions = (model.predict(alternatives_transformed) > 0.5).astype(int).flatten()
            # print(f"\n alternative_predictions: {alternative_predictions}")
        else:
            # Predict for traditional ML model
            original_prediction = model.predict(pd.DataFrame([original_instance]))[0]
            alternative_predictions = model.predict(alternatives)
        
        mismatched_instances = [
            {
                "alternative_instance": alternatives.iloc[j].to_dict(),
                "alternative_prediction": alternative_predictions[j]
            }
            for j in range(3) if alternative_predictions[j] != original_prediction
        ]

        if mismatched_instances:
            discriminatory_instances_count += 1
            if first_discrepancy_index is None:
                first_discrepancy_index = current_original_index


            for mismatch in mismatched_instances:
                output_data.append({
                    "original_index": current_original_index,
                    **original_instance.to_dict(),
                    "original_prediction": original_prediction,
                    **mismatch["alternative_instance"],
                    "alternative_prediction": mismatch["alternative_prediction"],
                })
                

    total_instances = len(df)
    discriminatory_ratio = (discriminatory_instances_count / total_instances) * 100

    print(f"Discriminatory Ratio: {discriminatory_ratio:.2f}%")
    if first_discrepancy_index:
        print(f"First discrepancy found at row: {first_discrepancy_index}")

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    print(f"Discriminatory instances saved to {output_file}")

    return {
        "Filename": os.path.basename(input_file),
        "Model_Name": os.path.basename(model_path),
        "Total_Instances": total_instances,
        "Discriminatory_Instances": discriminatory_instances_count,
        "Percentage_Discriminatory": round(discriminatory_ratio, 3),
        "First_Discrepancy": first_discrepancy_index if first_discrepancy_index else -1,
    }


def process_models(input_csv, models_dir, output_dir, results_dir):
    """
    Process models with the given input CSV and generate results.
    Args:
        input_csv (str): Path to the input CSV file.
        models_dir (str): Directory containing models.
        output_dir (str): Directory to save output CSV files.
        results_dir (str): Directory to save consolidated results.
    """
    
    # Get model paths
    model_paths = get_model_paths(models_dir)
    print(f"\nModel paths are: {model_paths}")
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load preprocessor for neural networks
    loaded_preprocessor = load_preprocessor()

    # Process each model
    for model_path in model_paths:
        print(f"\nProcessing model: {model_path}")
        is_nn = model_path.endswith(".h5")
        output_csv = os.path.join(output_dir, f"{os.path.basename(input_csv).split('.')[0]}_{os.path.basename(model_path).split('.')[0]}.csv")
        results = find_discriminatory_instances(
            input_file=input_csv,
            model_path=model_path,
            output_file=output_csv,
            preprocessor=loaded_preprocessor if is_nn else None,
            is_neural_network=is_nn
        )
        print(f"Results for {os.path.basename(model_path)}: {results}")
        # Append results to consolidated CSV, dynamically naming the filename according to the input filename
        append_results_to_csv(results, results_dir, input_csv)
    
    
    
if __name__ == "__main__":
    # input_csv = "../t_way_samples/compas_train_2_way_covering_array_bin_means_2024-12-25_21-14.csv"
    # input_csv = "../../Patel_Data/tWay_Concrete_TC/compas_AI360_Modified2_2way_concrete_TC_with_constraint.csv"
    input_csv = "../dataset/compas_train.csv"
    # input_csv = "../dataset/compas_test.csv"
    process_models(input_csv, models_dir, output_dir, results_dir)