import os
import sys

import numpy as np
from tensorflow.keras.models import load_model

from MRI.file_io import read_pickle
from MRI.plot_mri import plot_mri

def get_base_path():
    """
    Returns:
        str: The base path of the application.
    """
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


def print_index_ranges(y):
    """
    Print index ranges for ADHD and Healthy samples in the dataset.

    Args:
        y (numpy.ndarray): Array of labels where 1 indicates ADHD and 0 indicates healthy.
    """
    adhd_indices = np.where(y == 1)[0]
    healthy_indices = np.where(y == 0)[0]

    adhd_range = f"{adhd_indices[0]}-{adhd_indices[-1]}" if adhd_indices.size > 0 else "No indices"
    healthy_range = f"{healthy_indices[0]}-{healthy_indices[-1]}" if healthy_indices.size > 0 else "No indices"

    print(f"ADHD indices: {adhd_range}")
    print(f"Healthy indices: {healthy_range}")


def check_result(predictions, threshold=0.5):
    """
    Check the result of predictions and print the patient's condition and probability.

    Args:
        predictions (numpy.ndarray): Array of model predictions.
        threshold (float): Threshold to determine ADHD. Default is 0.5.

    Returns:
        tuple: A tuple containing the result ("ADHD" or "ZDROWY") and the probability (float).
    """
    mean_prediction = np.mean(predictions)
    if mean_prediction > threshold:
        result = "ADHD"
        probability = np.round(mean_prediction * 100, 2)
    else:
        result = "HEALTHY"
        probability = np.round((1 - mean_prediction) * 100, 2)

    print(f"Patient result: {result}, with probability: {probability}%")
    return result, probability


def predict_cnn(model_name, cnn_model, cnn_predict):

    """
    Predict using a pre-trained CNN model and evaluate the performance on a validation set.

    Args:
        model_name (str): The name of the model link (without extension).
        cnn_model (str): Directory path where the model link is located.
        cnn_predict (str): Directory path where the validation data files are located.
    """
    from MRI.config import CNN_SINGLE_INPUT_SHAPE_MRI
    try:
        model_path = os.path.join(cnn_model, f'{model_name}.keras')
        model = load_model(model_path)

        X = read_pickle(os.path.join(cnn_predict, f'X_pred_{model_name}.pkl'))
        y = read_pickle(os.path.join(cnn_predict, f'y_pred_{model_name}.pkl'))

    except OSError as e:
        print(f'Error loading model: {e}')
        return
    except Exception as e:
        print(f'Error loading data: {e}')
        return

    print_index_ranges(y)

    while True:
        try:
            image_number = int(input(f"Choose an image (0 - {X.shape[0] - 1}): "))
            if 0 <= image_number < X.shape[0]:
                break
            else:
                print("Enter an image number within the range")
        except ValueError:
            print("Enter an image number within the range")

    if y[image_number] == 1:
        print("You selected ADHD")
    else:
        print("You selected Healthy")

    plot_mri(X[image_number])

    _, accuracy = model.evaluate(X, y, verbose=0)
    img_for_predict = X[image_number].reshape(1, CNN_SINGLE_INPUT_SHAPE_MRI, CNN_SINGLE_INPUT_SHAPE_MRI, 1)
    predictions = model.predict(img_for_predict)

    check_result(predictions)
    print(f"Result on the entire validation set: {accuracy * 100:.2f} %")
