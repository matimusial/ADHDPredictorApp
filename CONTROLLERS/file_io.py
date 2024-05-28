import os
import pandas as pd
import mne
from scipy.io import loadmat, savemat


def readEEGRaw(path_or_folder):
    """
    Wczytuje surowe dane EEG / do poprawy i analizy

    Args:
    path_or_folder (str): Ścieżka do pliku .mat lub folderu z danymi.

    Returns:
    list lub tuple: Zawiera dane EEG w zależności od typu wejścia.
    """

    # Automatyczne wykrywanie typu wejścia
    if os.path.isfile(path_or_folder) and path_or_folder.endswith('.mat'):
        # Wersja 1: Wczytywanie pojedynczego pliku .mat
        mat_data = loadmat(path_or_folder, mat_dtype=True)
        file, _ = os.path.splitext(os.path.basename(path_or_folder))
        if file not in mat_data:
            raise KeyError(f"Klucz {file} nie znaleziony w pliku .mat.")

        return mat_data[file].T

    elif os.path.isdir(path_or_folder):
        # Wersja 2: Wczytywanie danych z folderu
        subfolders = ["ADHD", "CONTROL"]
        ADHD_DATA = []
        CONTROL_DATA = []

        # Sprawdzanie i konwersja plików CSV/EDF do formatu MATLAB
        for subfolder in subfolders:
            current_folder = os.path.join(path_or_folder, subfolder)
            if not os.path.isdir(current_folder):
                raise FileNotFoundError(f"Podfolder {current_folder} nie istnieje.")

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

        # Import plików .mat
        for subfolder in subfolders:
            current_folder = os.path.join(path_or_folder, subfolder)
            mat_files = [f for f in os.listdir(current_folder) if f.endswith('.mat')]

            for mat_file in mat_files:
                file_path = os.path.join(current_folder, mat_file)
                loaded_data = loadmat(file_path, mat_dtype=True)
                file_name, _ = os.path.splitext(mat_file)
                if file_name not in loaded_data:
                    raise KeyError(f"Klucz {file_name} nie znaleziony w pliku .mat {mat_file}.")

                if "ADHD" in subfolder:
                    ADHD_DATA.append(loaded_data[file_name].T)
                elif "CONTROL" in subfolder:
                    CONTROL_DATA.append(loaded_data[file_name].T)

        return ADHD_DATA, CONTROL_DATA

    else:
        raise ValueError("Podana ścieżka nie jest prawidłowym plikiem .mat ani folderem.")
