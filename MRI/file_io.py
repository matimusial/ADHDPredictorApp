import pickle
from sklearn.model_selection import train_test_split
from MRI.config import TEST_SIZE_MRI_CNN
import numpy as np


def read_pickle(filepath):
    """Reads data from a pickle link.

    Args:
        filepath (str): Path to the pickle link.

    Returns:
        object: Data read from the pickle link.
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def save_pickle(filepath, data):
    """Saves data to a pickle link.

    Args:
        filepath (str): Path to the pickle link.
        data (object): Data to be saved to the pickle link.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)


def prepare_for_cnn(adhd_data, control_data):
    """Prepares data for CNN.

    Args:
        adhd_data (list or np.array): Processed ADHD data.
        control_data (list or np.array): Processed control data.

    Returns:
        tuple: Split data ready for CNN (X_train, X_test, y_train, y_test).
    """
    from MRI.config import CNN_SINGLE_INPUT_SHAPE_MRI
    y_adhd = np.ones((len(adhd_data)))
    y_control = np.zeros((len(control_data)))
    y = np.hstack((y_adhd, y_control))

    X_adhd = np.reshape(adhd_data, (len(adhd_data), CNN_SINGLE_INPUT_SHAPE_MRI, CNN_SINGLE_INPUT_SHAPE_MRI, 1))
    X_control = np.reshape(control_data, (len(control_data), CNN_SINGLE_INPUT_SHAPE_MRI, CNN_SINGLE_INPUT_SHAPE_MRI, 1))
    X = np.vstack((X_adhd, X_control))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_MRI_CNN, shuffle=True)

    return X_train, X_test, y_train, y_test
