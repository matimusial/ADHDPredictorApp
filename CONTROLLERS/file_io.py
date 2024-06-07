import os
import pandas as pd
import mne
from scipy.io import loadmat, savemat



def read_eeg_raw(path_or_folder):
    """
    Loads raw EEG data for correction and analysis.

    Args:
        path_or_folder (str): Path to the .mat file or a folder with data.

    Returns:
        list or tuple: Contains EEG data depending on the input type.
    """

    adhd_files_count = 0
    control_files_count = 0
    table_structures = []

    if os.path.isfile(path_or_folder) and path_or_folder.endswith('.mat'):
        # Version 1: Loading a single .mat file
        mat_data = loadmat(path_or_folder, mat_dtype=True)
        file, _ = os.path.splitext(os.path.basename(path_or_folder))
        if file not in mat_data:
            raise KeyError(f"Key {file} not found in the .mat file.")

        return mat_data[file].T

    elif os.path.isdir(path_or_folder):
        # Version 2: Loading data from a folder
        subfolders = ["ADHD", "CONTROL"]
        ADHD_DATA = []
        CONTROL_DATA = []

        # Checking and converting CSV/EDF files to MATLAB format
        for subfolder in subfolders:
            current_folder = os.path.join(path_or_folder, subfolder)
            if not os.path.isdir(current_folder):
                raise FileNotFoundError(f"Subfolder {current_folder} does not exist.")

            for file in os.listdir(current_folder):
                if file.endswith('.csv') or file.endswith('.edf'):
                    csv_or_edf_file = os.path.join(current_folder, file)
                    mat_file_name = os.path.splitext(file)[0] + '.mat'

                    if file.endswith('.csv'):
                        df = pd.read_csv(csv_or_edf_file)
                        data_matrix = df.values
                        key_name = os.path.splitext(file)[0]
                        savemat(os.path.join(current_folder, mat_file_name), {key_name: data_matrix})

                    elif file.endswith('.edf'):
                        raw = mne.io.read_raw_edf(csv_or_edf_file, preload=True)
                        data = raw.get_data()
                        key_name = os.path.splitext(file)[0]
                        savemat(os.path.join(current_folder, mat_file_name), {key_name: data})

        # Importing .mat files
        for subfolder in subfolders:
            current_folder = os.path.join(path_or_folder, subfolder)
            mat_files = [f for f in os.listdir(current_folder) if f.endswith('.mat')]

            for mat_file in mat_files:
                file_path = os.path.join(current_folder, mat_file)
                loaded_data = loadmat(file_path, mat_dtype=True)
                file_name, _ = os.path.splitext(mat_file)
                if file_name not in loaded_data:
                    raise KeyError(f"Key {file_name} not found in .mat file {mat_file}.")

                data = loaded_data[file_name].T
                table_structure = {
                    'file': file_path,
                    'shape': data.shape,
                }
                table_structures.append(table_structure)

                if "ADHD" in subfolder:
                    ADHD_DATA.append(loaded_data[file_name].T)
                    adhd_files_count += 1
                elif "CONTROL" in subfolder:
                    CONTROL_DATA.append(loaded_data[file_name].T)
                    control_files_count += 1

        return ADHD_DATA, CONTROL_DATA, table_structures, adhd_files_count, control_files_count

    else:
        raise ValueError("The provided path is not a valid .mat file or folder.")

# Usage example:
# To load data from a .mat file
# eeg_data = read_eeg_raw('path/to/eeg_file.mat')
# print("EEG data from file:", eeg_data)

# To load data from a folder containing ADHD and CONTROL subfolders
# adhd_data, control_data = read_eeg_raw('path/to/eeg_data_folder')
# print("ADHD data:", adhd_data)
# print("Control data:", control_data)
