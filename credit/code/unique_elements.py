import pandas as pd
import argparse
import os

def find_unique_elements(file_path):
    """
    Find unique elements and their counts for each column in a CSV file.
    For continuous variables, data will be binned into 10 bins.
    
    :param file_path: Path to the CSV file
    :return: A string that contains unique elements and their counts for each column.
    """
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Initialize an empty string to store the results
        result_str = ""
        
        # Loop through each column in the DataFrame
        for column in df.columns:
            result_str += f"\n{column}\n"  # Add the column name to the result
            
            if pd.api.types.is_numeric_dtype(df[column]):  # Check if the column is numeric (continuous)
                # Bin the continuous variable into 10 bins
                binned_data = pd.cut(df[column], bins=20)
                value_counts = binned_data.value_counts().sort_index()
            else:
                # For non-continuous (categorical) variables, use value_counts() directly
                value_counts = df[column].value_counts()

            # Add each unique value (or bin) and its count to the result string
            for value, count in value_counts.items():
                result_str += f"{value}    {count}\n"
        
        return result_str
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Find unique elements and their counts in a CSV file.")
    parser.add_argument("file", help="Path to the CSV file.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Get the base name of the file (without the extension)
    base_filename = os.path.splitext(os.path.basename(args.file))[0]  # Strips the '.csv' extension
    output_filename = base_filename + "_results.csv"  # Appends '_results.csv' to the base file name
    
    # Call the function with the provided file path
    result = find_unique_elements(args.file)
    
    # If result is not None, save the output to a file
    if result is not None:
        # Write the result to a text file
        with open(output_filename, "w") as file:
            file.write(result)
        print(f"Results saved to {output_filename}")