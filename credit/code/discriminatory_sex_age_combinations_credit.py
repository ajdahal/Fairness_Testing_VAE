import os
import pickle
import random
import re
import joblib
import pandas as pd
from collections import defaultdict
import tensorflow as tf

base_dir = os.path.dirname("./")
output_dir = os.path.join(base_dir, "output")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")


# Fix random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_preprocessor(export_file_name="credit_preprocessor.pkl"):
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
    Get the most recent model paths sorted by date.
    """
    # Allowed model names
    allowed_models = {"logistic_regression", "nn", "random_forest", "svm"}
    
    # Initialize a defaultdict to group files by date
    models_by_date = defaultdict(list)
    
    # Regular expression to capture valid filenames and the date section
    pattern = re.compile(r"^credit__({})__(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}})\.(pkl|h5)$".format('|'.join(allowed_models)))
    
    # List all files in the models directory
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            model_name = match.group(1)  # Extract the model name
            date_section = match.group(2)  # Extract the date section
            full_path = os.path.join(models_dir, filename)
            models_by_date[date_section].append(full_path)

    # Get the most recent date section and corresponding files
    if models_by_date:
        latest_date_section = max(models_by_date.keys())  # Find the latest date section
        model_paths = sorted(models_by_date[latest_date_section])  # Get model paths for that date
    else:
        model_paths = []  # Handle case where no matching files are found
    print(f"\n model_paths: {model_paths}")
    return model_paths



def generate_combinations_from_csv(df, privileged_gender="Male", privileged_age_range=(26, 76)):
    """
    Generate original and alternative combinations of sensitive attributes for each row in a dataframe.
    Handles cases where column names differ (e.g., 'Attribute9' for gender and 'Attribute13' for age).
    """
    # Define unprivileged age range   -- in accordance to AI360
    unprivileged_age_range = list(range(19, 26))  # List of unprivileged ages (19 to 25 inclusive)
    privileged_age_range = list(range(26, 76))    # List of privileged ages (26 to 75 inclusive)

    # Define non-privileged genders
    non_privileged_genders = ['Female']
    
    print(f"\n privileged_gender: {privileged_gender}")
    print(f"\n non_privileged_genders: {non_privileged_genders}")
    
    print(f"\n unprivileged_age_range: {unprivileged_age_range}")
    print(f"\n privileged_age_range: {privileged_age_range}")
    
    combined_data = []
    for _, row in df.iterrows():
        original_instance = row.to_dict()
        combined_data.append(original_instance)

        current_gender = row['Attribute9']  # Use 'Attribute9' for gender
        current_age = row['Attribute13']    # Use 'Attribute13' for age

        # Determine if the row belongs to privileged or unprivileged groups
        is_privileged_gender = current_gender == privileged_gender
        is_privileged_age = current_age in privileged_age_range

        # Generate the combinations
        if is_privileged_gender and is_privileged_age:
            # Privileged gender, privileged age
            combined_data.append({**original_instance, 'Attribute9': random.choice(non_privileged_genders), 'Attribute13': random.choice(privileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': random.choice(non_privileged_genders), 'Attribute13': random.choice(unprivileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': privileged_gender, 'Attribute13': random.choice(unprivileged_age_range)})
        elif not is_privileged_gender and is_privileged_age:
            # Non-privileged gender, privileged age
            combined_data.append({**original_instance, 'Attribute9': privileged_gender, 'Attribute13': random.choice(privileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': random.choice(non_privileged_genders), 'Attribute13': random.choice(unprivileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': privileged_gender, 'Attribute13': random.choice(unprivileged_age_range)})
        elif not is_privileged_gender and not is_privileged_age:
            # Non-privileged gender, non-privileged age
            combined_data.append({**original_instance, 'Attribute9': privileged_gender, 'Attribute13': random.choice(unprivileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': random.choice(non_privileged_genders), 'Attribute13': random.choice(privileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': privileged_gender, 'Attribute13': random.choice(privileged_age_range)})
        elif is_privileged_gender and not is_privileged_age:
            # Privileged gender, non-privileged age
            combined_data.append({**original_instance, 'Attribute9': random.choice(non_privileged_genders), 'Attribute13': random.choice(privileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': random.choice(non_privileged_genders), 'Attribute13': random.choice(unprivileged_age_range)})
            combined_data.append({**original_instance, 'Attribute9': privileged_gender, 'Attribute13': random.choice(privileged_age_range)})
            
    return combined_data




def find_discriminatory_instances(input_file, model_path, output_file, preprocessor=None, is_neural_network=False):
    """
    Identify discriminatory instances by generating predictions for all combinations
    and comparing with the original instance. Handles both traditional ML models
    and neural networks.
    Returns: Results dictionary containing details of discriminatory analysis.
    """
    df = pd.read_csv(input_file)
    df.columns = ['Attribute1','Attribute2','Attribute3','Attribute4','Attribute5','Attribute6','Attribute7','Attribute8','Attribute9','Attribute10',
                  'Attribute11','Attribute12','Attribute13','Attribute14','Attribute15','Attribute16','Attribute17','Attribute18','Attribute19','Attribute20']
                
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


def fix_header_csv_file(input_csv):
    """
    Process the given CSV file:
    Returns: Path to the processed CSV file if processed, or the original input CSV file if no operation was performed.
    """
    # Target filename conditions
    target_filename1 = "../Patel_Data/tWay_Concrete_TC/GermanCredit_AI360_Modified_2way_concrete_TC_with_constraint.csv"
    target_filename2 = "../Patel_Data/Dataset/GermanCredit_AI360_Modified.csv"
    
    # Check if input file matches either target filename
    if input_csv in [target_filename1, target_filename2]:
        print(f"Input file matches one of the target filenames. Proceeding with processing: {input_csv}")
        
        # Problematic header order
        problematic_header_mapping = [
            "Attribute1", "Attribute2", "Attribute3", "Attribute4", "Attribute5", "Attribute6",
            "Attribute7", "Attribute8", "Attribute10", "Attribute11", "Attribute12", "Attribute13",
            "Attribute14", "Attribute15", "Attribute16", "Attribute17", "Attribute18", "Attribute19",
            "Attribute20", "Attribute9"
        ]
        
        # Correct header order
        correct_header_order = [
            "Attribute1", "Attribute2", "Attribute3", "Attribute4", "Attribute5", "Attribute6",
            "Attribute7", "Attribute8", "Attribute9", "Attribute10", "Attribute11", "Attribute12",
            "Attribute13", "Attribute14", "Attribute15", "Attribute16", "Attribute17", "Attribute18",
            "Attribute19", "Attribute20"
        ]
        
        # Load the CSV
        data = pd.read_csv(input_csv)
        data = data.drop(columns=['GoodCredit']) if 'GoodCredit' in data.columns else data

        
        print(f"\nBefore reordering: data.columns: {list(data.columns)}")
        
        # Assign problematic headers to the columns
        data.columns = problematic_header_mapping
        
        # Reorder the columns
        data = data[correct_header_order]
        
        print(f"\nAfter reordering: data.columns: {list(data.columns)}")
        print(f"\nPreview of data:\n{data.head(3)}")
        
        # Generate output file path
        base_filename = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = f"../output/{base_filename}_columns_reordered.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)  # Ensure the output directory exists
        data.to_csv(output_csv, index=False)
        print(f"Reordered CSV saved as: {output_csv}")
        
        # Return the new file path
        return output_csv
    else:
        print("File name doesn't match any target filenames. No operation performed.")
        # Return the original input file path
        return input_csv



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
    # input_csv = "../t_way_samples/adult_train_2_way_covering_array_bin_means_2024-12-25_21-13.csv"
    input_csv = "../../Patel_Data/tWay_Concrete_TC/GermanCredit_AI360_Modified_2way_concrete_TC_with_constraint.csv"
    process_models(input_csv, models_dir, output_dir, results_dir)