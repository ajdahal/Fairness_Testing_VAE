# Fairness_Testing_VAE
This repo contains the necessary code for generation of samples for Fairness Testing using t_way sampling in the latent space of a Variational AutoEncoder.<br>
1. requirements.txt: contains the necessary libraries required to run the code.<br>
2. utils: contains the utility scripts for downloading necessary data to compare the results and add some functions to tvae.<br>
3. dataset_name:
    -    dataset_name_driver.py: contains code to obtain results of fairness violations for t = 2.
    -    find_discrimination_for_other_files.py: contains code to obtain results of fairness violations for higher order t-way combinations.
    -    code: contains necessary scripts for running the project.
    -    dataset: contains dataset files.
    -    models: contains the VAE model and machine learning models (random_forest, logistic_regression, svm, nn).
    -    results: contains the results ( ATN, ratio of discriminatory_instances, First Discriminatory Instance ).


Steps to run the code:
1. Create a virtual environment in python before running the code.
2. Install the necessary libraries using requirements.txt.
3. Run "download_baseline_and_replace_files.py" from utils folder.
    - Downloads t-way test cases in input space from Patel, et al.
    - Replaces the files (ctgan.py, tvae.py) in SDV using the files in utils.
4. Select the folder (the folders are named according to the name of datasets as Adult, COMPAS or Credit),  and run the file "dataset_name_driver.py".
    - The output is results of fairness violations by t-way combination for t = 2 in the latent space and input space for four machine learning models.
5. Run the file "find_discrimination_for_other_files.py" for results related to fairness violations caused by higher order t-way combinations.
