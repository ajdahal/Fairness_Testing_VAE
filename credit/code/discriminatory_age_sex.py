import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rdt import HyperTransformer
from datetime import datetime


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
    # Set the seed for PyTorch (CPU)
    torch.manual_seed(seed)
    # Set the seed for PyTorch (GPU, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set the seed for rdt HyperTransformer
    ht = HyperTransformer()
    ht.random_state = seed
    print(f"Seed initialized to: {seed}")




# Define the MLPClassifier model
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[1], output_size)
        )
    
    def forward(self, x):
        return self.network(x)


def load_model(model_path, input_size, hidden_sizes, output_size):
    model = MLPClassifier(input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def make_predictions(model, data_loader):
    all_predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs[0])
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            all_predictions.extend(predicted_classes.numpy())
    return all_predictions


def flip_gender_column(dataframe):
    dataframe = dataframe.copy()
    dataframe['Attribute9'] = dataframe['Attribute9'].apply(lambda x: 'Male' if x == 'Female' else 'Female')
    return dataframe


def flip_age(dataframe):
    column_name='Attribute13'
    dataframe = dataframe.copy()
    unique_values = dataframe[column_name].unique().tolist()
    # Function to choose a different value
    def choose_different(current_value):
        other_values = [value for value in unique_values if value != current_value]
        return random.choice(other_values)
    dataframe[column_name] = dataframe[column_name].apply(choose_different)
    return dataframe


def find_first_discrepancy(predictions, flipped_predictions, description):
    for idx, (pred, flipped_pred) in enumerate(zip(predictions, flipped_predictions)):
        if pred != flipped_pred:
            print(f"First {description} discrepancy found at row: {idx + 1}")
            return idx + 1
    return None


def process_file(file_path, model, ht, input_size, hidden_sizes, output_size):
    
    # finds the percentage of discriminatory instances
    
    df = pd.read_csv(file_path)
    X_test = df.drop('class', axis=1) if 'class' in df.columns else df
    print(f"Discriminatory_sex_age | X_test: {X_test[0:5]}")
    columns_to_keep = ['Attribute1','Attribute2','Attribute3','Attribute4','Attribute5','Attribute6','Attribute7','Attribute8','Attribute9','Attribute10','Attribute11','Attribute12','Attribute13','Attribute14','Attribute15','Attribute16','Attribute17','Attribute18','Attribute19','Attribute20']
    X_test = X_test[columns_to_keep]
    X_test_gender_flipped = flip_gender_column(X_test)
    X_test_age_flipped = flip_age(X_test)
    
    transformed_data = ht.transform(X_test)
    # print(f"transformed_data's shape: {transformed_data.shape} | transformed_data: {transformed_data[0:5]}")
    transformed_data_gender_flipped = ht.transform(X_test_gender_flipped)
    transformed_data_age_flipped = ht.transform(X_test_age_flipped)
    
    input_tensor = torch.tensor(transformed_data.values, dtype=torch.float32)
    input_tensor_gender_flipped = torch.tensor(transformed_data_gender_flipped.values, dtype=torch.float32)
    input_tensor_age_flipped = torch.tensor(transformed_data_age_flipped.values, dtype=torch.float32)

    dataset = TensorDataset(input_tensor)
    dataset_gender_flipped = TensorDataset(input_tensor_gender_flipped)
    dataset_age_flipped = TensorDataset(input_tensor_age_flipped)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    data_loader_gender_flipped = DataLoader(dataset_gender_flipped, batch_size=32, shuffle=False)
    data_loader_age_flipped = DataLoader(dataset_age_flipped, batch_size=32, shuffle=False)

    predictions = np.array(make_predictions(model, data_loader))
    predictions_gender_flipped = np.array(make_predictions(model, data_loader_gender_flipped))
    predictions_age_flipped = np.array(make_predictions(model, data_loader_age_flipped))

    total_instances = len(df)
    print(f"\n predictions: {predictions}")
    print(f"\n predictions_gender_flipped: {predictions_gender_flipped}")
    
    discriminatory_instances_gender = np.sum(predictions != predictions_gender_flipped)
    discriminatory_instances_age = np.sum(predictions != predictions_age_flipped)

    discriminatory_percentage_gender = (discriminatory_instances_gender / total_instances) * 100
    discriminatory_percentage_age = (discriminatory_instances_age / total_instances) * 100

    # Identify the first instances
    first_gender_discrepancy = find_first_discrepancy(predictions, predictions_gender_flipped, "gender-flipped")
    first_age_discrepancy = find_first_discrepancy(predictions, predictions_age_flipped, "age-flipped")
    
    return {
        "Filename": os.path.basename(file_path),
        "Total_Instances": int(total_instances) if total_instances is not None else 0,  # Default to 0 if None
        "Discriminatory_Instances_Sex": int(discriminatory_instances_gender) if discriminatory_instances_gender is not None else 0,  # Default to 0 if None
        "Percentage_Discriminatory_Sex": round(discriminatory_percentage_gender, 3) if discriminatory_percentage_gender is not None else 0.0,  # Default to 0.0 if None
        "First_Discrepancy_Sex": int(first_gender_discrepancy) if first_gender_discrepancy is not None else -1,  # Default to -1 if None
        "Discriminatory_Instances_Age": int(discriminatory_instances_age) if discriminatory_instances_age is not None else 0,  # Default to 0 if None
        "Percentage_Discriminatory_Age": round(discriminatory_percentage_age, 3) if discriminatory_percentage_age is not None else 0.0,  # Default to 0.0 if None
        "First_Discrepancy_Age": int(first_age_discrepancy) if first_age_discrepancy is not None else -1,  # Default to -1 if None
}





def process_folder(folder_path, model_path, ht_path, results_folder, input_size=53, hidden_sizes=[256, 128], output_size=2):
    initialize_seed_from_env()
    
    with open(ht_path, 'rb') as f:
        ht = pickle.load(f)
    model = load_model(model_path, input_size, hidden_sizes, output_size)
    os.makedirs(results_folder, exist_ok=True)
    results = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            result = process_file(file_path, model, ht, input_size, hidden_sizes, output_size)
            result["Model_Name"] = os.path.basename(model_path)
            result["Hypertransformer_Name"] = os.path.basename(ht_path)
            results.append(result)

    results = sorted(results, key=lambda x: x["Filename"])
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(results_folder, f"discriminatory_instances_results_{current_timestamp}.csv")
    results_df.to_csv(output_csv, index=False)
    return output_csv


def main():
    initialize_seed_from_env()
    
    parser = argparse.ArgumentParser(description="Find discriminatory instances for gender and race in CSV files.")
    parser.add_argument("--folder_path", type=str, default="../dataset/",help="Path to the folder containing input CSV files.")
    parser.add_argument("--model_path", type=str, default="../models/black_box_model_vae_essential_columns_credit.pth", help="Path to the trained model file.")
    parser.add_argument("--ht_path", type=str, default="../models/black_box_model_train_for_VAE_credit_hypertransformer.pkl", help="Path to the HyperTransformer pickle file.")
    parser.add_argument("--results_folder", type=str, default="results", help="Folder to save the results CSV.")
    args = parser.parse_args()
    
    output_csv = process_folder(args.folder_path, args.model_path, args.ht_path, args.results_folder)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()