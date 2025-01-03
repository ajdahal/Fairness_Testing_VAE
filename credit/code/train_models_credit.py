# Necessary Imports
import os
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import json
import glob
import random
import joblib
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import load_model as keras_load_model
from keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, StringLookup, CategoryEncoding, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers



# Define file paths and directories
base_dir = os.path.dirname("./")
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
t_way_samples_dir = os.path.join(base_dir,"t_way_samples")
train_file_path = os.path.join(dataset_dir, "credit_train.csv")
test_file_path = os.path.join(dataset_dir, "credit_test.csv")
processor_export_file_path = os.path.join(models_dir, "credit_preprocessor.pkl")


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


# Data Preprocessing
def get_preprocessor(numerical_features, categorical_features):
    
    # Numeric transformations
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())  # Scale numerical features to the range [0, 1]
    ])
    
    # Categorical transformations
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Handle unknown categories during transformation
    ])
    
    # Combined transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


# Train Logistic Regression
def train_logistic_regression(preprocessor, X_train, y_train, model_path):
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    clf.fit(X_train, y_train)
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)
    return model_path



# Train Random Forest
def train_random_forest(preprocessor, X_train, y_train, model_path):
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
    ])
    clf.fit(X_train, y_train)
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)
    return model_path




# Train Support Vector Machine
def train_svm(preprocessor, X_train, y_train, model_path):
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(gamma='auto', probability=True,random_state=42))                   # can be used to get the probabilities of the data instances 
    ])
    clf.fit(X_train, y_train)
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)
    return model_path




def train_nn(X_train, y_train, x_test, y_test, model_path, numerical_cols, categorical_cols, preprocessor):
    """
    Train a neural network model with preprocessing performed beforehand.
    Returns, Path to the saved model.
    """
    
    # Transform the data
    preprocessor.fit(X_train)
    
    # Export the preprocessor to a file
    joblib.dump(preprocessor, processor_export_file_path)
    print(f"Preprocessor exported to {processor_export_file_path}")
    
    X_train_transformed = preprocessor.transform(X_train)
    print(f"\n X_train_transformed: {X_train_transformed[0:5]} \n\n")               #  this is a sparse matrix
    
    # Define the model
    input_layer = Input(shape=(X_train_transformed.shape[1],), name='input_layer')
    x = Dense(20, activation='relu', kernel_regularizer=regularizers.l1(0.001))(input_layer)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_transformed, y_train, validation_split=0.2, epochs=100, verbose=0, class_weight={0: 1, 1: 2},shuffle=False)

    # Save the model in HDF5 format
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Transform the data
    X_test_transformed = preprocessor.transform(x_test)
    
    # Debug: Print shapes and column names of transformed data
    print(f"\n X_train_transformed shape: {X_train_transformed.shape}")
    print(f"\n X_test_transformed shape: {X_test_transformed.shape}")
    
    # Ensure column alignment
    if X_test_transformed.shape[1] != X_train_transformed.shape[1]:
        raise ValueError(f"Mismatch in feature dimensions: Train ({X_train_transformed.shape[1]}) vs Test ({X_test_transformed.shape[1]})")
    
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    # Predict probabilities
    y_pred_probs = model.predict(X_test_transformed)
    print(f"\n y_pred_probs: {y_pred_probs}")
    # Convert probabilities to class labels
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    # Calculate accuracy
    nn_model_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n accuracy score of neural network: {nn_model_accuracy}")
    return nn_model_accuracy, model_path 




# Load Model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Evaluate Test Accuracy
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy



def write_nn_predictions(model_path, input_data_file, output_file_path, preprocessor):
    """
    Write predictions for each instance in the input data file.
    """
    # Load the input data
    input_data = pd.read_csv(input_data_file)
    # Transform the input data
    X_transformed = preprocessor.transform(input_data)
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    # Predict probabilities
    y_pred_probs = model.predict(X_transformed)
    # Add predictions rounded to 2 decimal points
    input_data['Prediction_Prob'] = np.round(y_pred_probs.flatten(), 2)
    # Add binary predictions (0 or 1) based on the threshold of 0.5
    input_data['Prediction'] = (y_pred_probs > 0.5).astype(int).flatten()
    # Write the predictions to the output file
    input_data.to_csv(output_file_path, index=False)
    print(f"Predictions written to {output_file_path}")



def write_notnn_predictions(model_path):
    
    # Dynamically obtain the input file
    query_file_name = next(iter(glob.glob(os.path.join(t_way_samples_dir, "credit_train_2_way_covering_array_bin_means*.csv"))), None)
    if not query_file_name:
        print("No matching input file found in the specified directory.")
        return
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Generate the output file name
    output_file_path = os.path.join(
        t_way_samples_dir, 
        f"predictions_{os.path.splitext(os.path.basename(query_file_name))[0]}_{model_name}.csv"
    )
    
    # Load the input data
    try:
        input_data = pd.read_csv(query_file_name)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Remove the 'class' column if it exists
    if 'class' in input_data.columns:
        print("Removing 'class' column from input data...")
        input_data = input_data.drop(columns=['class'])

    # Load the model
    try:
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Generate predictions
    try:
        predictions = loaded_model.predict(input_data)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Add predictions to the dataframe
    input_data["Prediction"] = predictions

    
    # Write the dataframe to the output CSV file
    try:
        input_data.to_csv(output_file_path, index=False)
        print(f"Predictions successfully written to {output_file_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")



# Main Function
def main(train_file_path, test_file_path, model_dir):
    
    x_train = pd.read_csv(train_file_path)
    x_test = pd.read_csv(test_file_path)
    
    # Ensure target variables are 1D arrays and adjust labels
    y_train = pd.read_csv(train_file_path, usecols=['class']).squeeze() - 1
    y_test = pd.read_csv(test_file_path, usecols=['class']).squeeze() - 1
    
    # Define numerical and categorical features
    numerical_cols = ['Attribute2', 'Attribute5', 'Attribute8', 'Attribute11', 'Attribute16', 'Attribute18', 'Attribute13']
    categorical_cols = list(set(x_train.columns) - set(numerical_cols) - {'class'})
    
    
    # Get preprocessor
    preprocessor = get_preprocessor(numerical_cols, categorical_cols)
    
    
    # Generate a list of model paths
    model_name = ["_logistic_regression__", "_random_forest__", "_svm__"]
    # Assume `model_name` is a list, e.g., ["logistic_regression", "random_forest", "svm"]
    model_paths = [
        os.path.join(models_dir, "credit_" + model + ".pkl")
        for model in model_name  # Iterate over each model name in the list
    ]
    
    # Train and save Logistic Regression
    logistic_model_path = train_logistic_regression(preprocessor, x_train, y_train, model_paths[0])
    logistic_model = load_model(logistic_model_path)
    logistic_accuracy = evaluate_model(logistic_model, x_test, y_test)
    print(f"Logistic Regression model saved to: {logistic_model_path}, Test Accuracy: {100 * logistic_accuracy:.2f}")
    
    
    # Train and save Random Forest
    random_forest_model_path = train_random_forest(preprocessor, x_train, y_train, model_paths[1])
    random_forest_model = load_model(random_forest_model_path)
    random_forest_accuracy = evaluate_model(random_forest_model, x_test, y_test)
    print(f"Random Forest model saved to: {random_forest_model_path}, Test Accuracy: {100 * random_forest_accuracy:.2f}")
    
    
    # Train and save SVM
    svm_model_path = train_svm(preprocessor, x_train, y_train, model_paths[2])
    svm_model = load_model(svm_model_path)
    svm_accuracy = evaluate_model(svm_model, x_test, y_test)
    print(f"SVM model saved to: {svm_model_path}, Test Accuracy: {100 * svm_accuracy:.2f}")
    
    
    write_notnn_predictions(logistic_model_path)
    write_notnn_predictions(random_forest_model_path)
    write_notnn_predictions(svm_model_path)
    
    
    # Train and save NN
    model_path_nn = os.path.join(models_dir, "credit_" + "_nn__" + ".h5")
    nn_model_accuracy, nn_model_path = train_nn(x_train, y_train, x_test, y_test, model_path_nn, numerical_cols, categorical_cols, preprocessor)
    print(f"NN model saved to: {nn_model_path}, Test Accuracy: {100 * nn_model_accuracy:.2f}")
    
    
    # Write the predictions to this file
    query_file_name = next(iter(glob.glob(os.path.join(t_way_samples_dir, "credit_train_2_way_covering_array_bin_means*.csv"))), None) 
    # Generate the output file name
    output_file_path = os.path.join(t_way_samples_dir, f"predictions_{os.path.splitext(os.path.basename(query_file_name))[0]}_{os.path.splitext(os.path.basename(model_path_nn))[0]}.csv")
    write_nn_predictions(nn_model_path, query_file_name, output_file_path, preprocessor)
    
    
# Entry Point
if __name__ == "__main__":
    # Standalone script behavior
    main(train_file_path, test_file_path, models_dir)