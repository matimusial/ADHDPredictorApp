import os
import scipy.io
import pandas as pd
import pyedflib
import numpy as np


# function not used in our project (one-time use)


def convert_mat_to_csv_and_edf(input_directory, csv_output_directory, edf_output_directory, edf_channel_labels, sample_rate=128):
    """
    Converts .mat files from the input directory to .csv and .edf files, saving them in the respective output directories.
    Additionally, checks if .edf files have correct number of EEG channels and other required specifications.

    Args:
        input_directory (str): Path to the directory containing .mat files.
        csv_output_directory (str): Path to the directory where .csv files will be saved.
        edf_output_directory (str): Path to the directory where .edf files will be saved.
        edf_channel_labels (list of str): List of labels for the EEG channels.
        sample_rate (int, optional): Sample rate of the EEG data. Default is 128 Hz.
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
            eeg_data = mat_data['eeg'].T
            assert eeg_data.shape[0] == len(edf_channel_labels), "Incorrect number of EEG channels"

            n_channels = len(edf_channel_labels)
            n_samples = eeg_data.shape[1]

            edf_file_path = os.path.join(edf_output_directory, f'{os.path.splitext(mat_file)[0]}.edf')
            with pyedflib.EdfWriter(edf_file_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
                channel_info = []
                for i, label in enumerate(edf_channel_labels):
                    ch_dict = {
                        'label': label,
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
        except Exception as e:
            print(f"Error processing {mat_file}: {e}")

    print("Conversion to .csv and .edf completed.")

# Usage example:
# convert_mat_to_csv_and_edf('input_dir', 'output_csv_dir', 'output_edf_dir', ['Fz', 'Cz', 'Pz', 'C3', ...], 128)
electrode_positions = ["Fz", "Cz", "Pz", "C3", "T3", "C4", "T4", "Fp1", "Fp2", "F3", "F4", "F7", "F8", "P3", "P4", "T5", "T6", "O1", "O2"]