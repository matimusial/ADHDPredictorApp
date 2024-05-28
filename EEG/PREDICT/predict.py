import os
import numpy as np

from keras.models import load_model

from EEG.data_preprocessing import filter_eeg_data, clip_eeg_data, normalize_eeg_data
from EEG.file_io import read_pickle, split_into_frames

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_index_ranges(y):
    """
    Print index ranges for ADHD and Healthy samples in the dataset.

    Args:
        y (numpy.ndarray): Array of labels where 1 indicates ADHD and 0 indicates healthy.
    """
    adhd_indices = np.where(y == 1)[0]
    healthy_indices = np.where(y == 0)[0]

    adhd_range = f"{adhd_indices[0]}-{adhd_indices[-1]}" if adhd_indices.size > 0 else "Brak indeksów"
    healthy_range = f"{healthy_indices[0]}-{healthy_indices[-1]}" if healthy_indices.size > 0 else "Brak indeksów"

    print(f"Indeksy ADHD: {adhd_range}")
    print(f"Indeksy Zdrowe: {healthy_range}")


def check_result(predictions, threshold=0.5):
    """
    Check the result of predictions against a threshold to determine ADHD or Healthy status.

    Args:
        predictions (numpy.ndarray): Array of prediction probabilities from the model.
        threshold (float): The threshold probability to classify as ADHD. Default is 0.5.

    Returns:
        tuple: A tuple containing the result ("ADHD" or "ZDROWY") and the probability percentage.
    """
    mean = np.mean(predictions)
    if mean > threshold:
        result = "ADHD"
        prob = np.round(mean * 100, 2)
    else:
        result = "ZDROWY"
        prob = np.abs(np.round((1 - mean) * 100, 2))

    print(f"Wynik pacjenta: {result}, z prawdopodobieństwem: {prob}%")

    return result, prob


def predict(MODEL_NAME, model_path, pickle_path):
    """
    Predict the ADHD or Healthy status of a patient using a pre-trained model.

    Args:
        MODEL_NAME (str): The name of the model to load.
        model_path (str): The directory path where the model is stored.
        pickle_path (str): The directory path where the pickled data files are stored.
    """
    try:
        model_full_path = os.path.join(model_path, f"{MODEL_NAME}.keras")
        model = load_model(model_full_path)

        X_path = os.path.join(pickle_path, f'X_pred_{MODEL_NAME}.pkl')
        y_path = os.path.join(pickle_path, f'y_pred_{MODEL_NAME}.pkl')

        X = read_pickle(X_path)
        y = read_pickle(y_path)
    except Exception as e:
        print(f'Błędna ścieżka do modelu: {e}')
        return

    print_index_ranges(y)

    while True:
        try:
            patient_number = int(input("Wybierz numer pacjenta: "))
            if 0 <= patient_number < len(X):
                break
            else:
                print("Wpisz numer pacjenta w zakresie")
        except Exception as e:
            print("Wpisz numer pacjenta w zakresie")

    if y[patient_number] == 1:
        print("Wybrałeś ADHD")
    elif y[patient_number] == 0:
        print("Wybrałeś Zdrowy")

    DATA = X[patient_number]

    DATA_FILTERED = filter_eeg_data(DATA)
    DATA_CLIPPED = clip_eeg_data(DATA_FILTERED)
    DATA_NORMALIZED = normalize_eeg_data(DATA_CLIPPED)
    DATA_FRAMED = split_into_frames(np.array(DATA_NORMALIZED))

    predictions = model.predict(DATA_FRAMED)
    check_result(predictions)
