import os
import scipy.io
import pandas as pd
import pyedflib
import numpy as np

def convert_mat_to_csv_and_edf(input_directory, csv_output_directory, edf_output_directory):
    """
    Converts .mat files from the input directory to .csv and .edf files, saving them in the respective output directories.

    Args:
        input_directory (str): Path to the directory containing .mat files.
        csv_output_directory (str): Path to the directory where .csv files will be saved.
        edf_output_directory (str): Path to the directory where .edf files will be saved.
    """

    os.makedirs(csv_output_directory, exist_ok=True)
    os.makedirs(edf_output_directory, exist_ok=True)

    mat_files = [f for f in os.listdir(input_directory) if f.endswith('.mat')]

    for mat_file in mat_files:
        try:
            mat_path = os.path.join(input_directory, mat_file)
            mat_data = scipy.io.loadmat(mat_path)

            # Save as .csv
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray):
                    df = pd.DataFrame(value)
                    csv_file = os.path.join(csv_output_directory, f'{os.path.splitext(mat_file)[0]}_{key}.csv')
                    df.to_csv(csv_file, index=False)

            # Save as .edf
            edf_file = os.path.join(edf_output_directory, f'{os.path.splitext(mat_file)[0]}.edf')
            signal_headers = []
            signals = []
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray):
                    signal_headers.append({'label': key})
                    signals.append(value.flatten())

            num_channels = len(signal_headers)
            if num_channels > 0:
                with pyedflib.EdfWriter(edf_file, num_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
                    f.setSignalHeaders(signal_headers)
                    f.writeSamples(signals)
        except Exception as e:
            print(f"Błąd w przetwarzaniu {mat_file}: {e}")

    print("Konwersja zakończona.")
