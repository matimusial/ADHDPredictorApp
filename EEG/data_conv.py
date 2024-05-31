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
        except Exception as e:
            print(f"Błąd w przetwarzaniu {mat_file}: {e}")

    print("Konwersja zakończona.")

# do edf
import scipy.io
import numpy as np
import pyedflib

name = ['v41p', 'v42p', 'v43p', 'v44p', 'v45p']
for i in name:
    # Załaduj plik .mat
    mat_data = scipy.io.loadmat(f'CONTROLLERS/INPUT_DATA/EEG/MAT/CONTROL/{i}.mat')

    eeg_data = mat_data[f'{i}'].T

    # Sprawdź wymiary danych
    assert eeg_data.shape[0] == 19, "Dane muszą zawierać 19 kanałów EEG"

    # Parametry sygnału
    n_channels = 19  # liczba kanałów
    n_samples = eeg_data.shape[1]  # liczba próbek

    # Etykiety kanałów
    channel_labels = ['Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'P3', 'P4', 'T5',
                      'T6', 'O1', 'O2']
    sample_rate = 128

    # Stwórz nowy plik EDF
    with pyedflib.EdfWriter(f'{i}.edf', n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        channel_info = []
        for i in range(n_channels):
            ch_dict = {
                'label': channel_labels[i],
                'dimension': 'uV',
                'sample_rate': sample_rate,
                'physical_min': np.min(eeg_data[i]),
                'physical_max': np.max(eeg_data[i]),
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': '',
                'prefilter': ''
            }
            channel_info.append(ch_dict)

        f.setSignalHeaders(channel_info)
        f.writeSamples(eeg_data)

    print("Plik EDF został zapisany.")