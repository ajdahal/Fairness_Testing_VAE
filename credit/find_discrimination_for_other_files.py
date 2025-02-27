import os
import sys

# Add the 'code' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from discriminatory_sex_age_combinations_credit import fix_header_csv_file, process_models

base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, "models")
output_dir = os.path.join(base_dir, "output")
results_dir = os.path.join(base_dir, "results")
dataset_dir = os.path.join(base_dir, "dataset")

t_way_dir = "t_way_samples"
csv_files = ['credit_train_3_way_covering_array_bin_means_2024-12-31_12-04.csv', 'credit_train_4_way_covering_array_bin_means_2024-12-31_12-04.csv']

csv_paths = [os.path.join(t_way_dir, f) for f in csv_files]

for t_way_file in csv_paths:
    process_models(t_way_file, models_dir, output_dir, results_dir)


#dataset_files  = ["credit_train.csv", "credit_test.csv"]
#dataset_csv_files = [os.path.join(dataset_dir, f) for f in dataset_files]

#for dataset_file in dataset_csv_files:
#    process_models(dataset_file, models_dir, output_dir, results_dir)
