import os
import shutil
import subprocess

def manage_repo(repo_url, destination_folder, folders_to_save):
    """
    Clone a GitHub repository, keep specified folders in the destination folder's root, and delete the rest.

    Args:
        repo_url (str): The GitHub repository URL.
        destination_folder (str): The folder where the repository should be cloned.
        folders_to_save (list): List of folder paths (relative to repo root) to keep.
    """
    try:
        # Step 1: Clone the repository
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        subprocess.run(["git", "clone", repo_url, destination_folder], check=True)
        print(f"Repository cloned to: {destination_folder}")

        # Step 2: Prepare to move necessary folders to the root
        absolute_folders_to_save = [
            os.path.join(destination_folder, folder) for folder in folders_to_save
        ]

        # Create a temporary list to store paths of folders moved to root
        temp_moved_folders = []

        for folder in absolute_folders_to_save:
            if os.path.exists(folder):
                # Move each folder to the root level of destination_folder
                new_path = os.path.join(destination_folder, os.path.basename(folder))
                shutil.move(folder, new_path)
                temp_moved_folders.append(new_path)
                print(f"Moved: {folder} -> {new_path}")
            else:
                print(f"Warning: {folder} does not exist in the repository.")

        # Step 3: Delete everything else in the destination folder
        for item in os.listdir(destination_folder):
            item_path = os.path.join(destination_folder, item)
            if item_path not in temp_moved_folders:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
                elif os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")

        print("\nCleanup complete. Remaining items in destination folder:")
        for item in os.listdir(destination_folder):
            print(f"  - {item}")

    except Exception as e:
        print(f"Error managing repository: {e}")
        
        from repo_manager import manage_repo 

if __name__ == "__main__":
    # Input from the user or pre-configured
    repo_url = "https://github.com/patelankitar/tWayFairnessTesting.git"
    destination_folder = "../Patel_Data"
    folders_to_save = ["Dataset", "Data/tWay_Concrete_TC"]

    # Call the manage_repo function
    manage_repo(repo_url, destination_folder, folders_to_save)