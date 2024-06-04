import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from EEG.config import EEG_SIGNAL_FRAME_SIZE


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


def split_into_frames(data):
    """Splits the data into frames of EEG_SIGNAL_FRAME_SIZE.

    Args:
        data (numpy.ndarray): The EEG data to be split.

    Returns:
        numpy.ndarray: The framed EEG data.

    Raises:
        ValueError: If the number of samples is less than the frame size.
    """
    try:
        if data.shape[1] < EEG_SIGNAL_FRAME_SIZE:
            raise ValueError("The number of samples is less than the frame size.")

        num_frames = data.shape[1] // EEG_SIGNAL_FRAME_SIZE
        framed_data = np.zeros((num_frames, data.shape[0], EEG_SIGNAL_FRAME_SIZE))

        for i in range(num_frames):
            framed_data[i, :, :] = data[:, i * EEG_SIGNAL_FRAME_SIZE: (i + 1) * EEG_SIGNAL_FRAME_SIZE]

        return framed_data
    except Exception as e:
        print(f"Error splitting data into frames: {e}")
        return None


def make_pred_data(adhd_data, control_data):
    """Updates the dataset by removing predictive data and returning it.

    Args:
        adhd_data (list of numpy.ndarray): List of EEG data arrays for individuals with ADHD.
        control_data (list of numpy.ndarray): List of EEG data arrays for the control group.

    Returns:
        tuple: Updated datasets and predictive data.
    """
    try:
        adhd_index = np.random.choice(range(0, len(adhd_data)), size=4, replace=False)
        control_index = np.random.choice(range(0, len(control_data)), size=4, replace=False)

        adhd_pred = []
        for i in adhd_index:
            adhd_pred.append(adhd_data[i])

        control_pred = []
        for i in control_index:
            control_pred.append(control_data[i])

        adhd_data_tt = []
        for i in range(len(adhd_data)):
            if i not in adhd_index:
                adhd_data_tt.append(adhd_data[i])

        control_data_tt = []
        for i in range(len(control_data)):
            if i not in control_index:
                control_data_tt.append(control_data[i])

        y_adhd_pred = np.ones(len(adhd_pred))
        y_control_pred = np.zeros(len(control_pred))
        y_pred = np.hstack((y_adhd_pred, y_control_pred))
        x_pred = adhd_pred + control_pred

        return adhd_data_tt, control_data_tt, x_pred, y_pred

    except Exception as e:
        print(f"Error in function update_data: {e}")
        return None, None, None, None


def prepare_for_cnn(adhd_data_tt, control_data_tt):
    """Prepares EEG data for training and testing CNN.

    Args:
        adhd_data_tt (list of numpy.ndarray): Updated list of EEG data arrays for individuals with ADHD.
        control_data_tt (list of numpy.ndarray): Updated list of EEG data arrays for the control group.

    Returns:
        tuple: Prepared data for training and testing the CNN.
    """

    try:
        adhd_framed_list = []
        for patient_data in adhd_data_tt:
            adhd_framed_list.append(split_into_frames(patient_data))

        control_framed_list = []
        for patient_data in control_data_tt:
            control_framed_list.append(split_into_frames(patient_data))

        adhd_framed = np.concatenate(adhd_framed_list, axis=0)
        control_framed = np.concatenate(control_framed_list, axis=0)

        y_adhd = np.ones(adhd_framed.shape[0])
        y_control = np.zeros(control_framed.shape[0])

        x = np.concatenate((control_framed, adhd_framed))
        y = np.concatenate((np.array(y_control), np.array(y_adhd)))

        x_4d = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
        x_train, x_test, y_train, y_test = train_test_split(x_4d, y, test_size=0.2, shuffle=True)

        return x_train, y_train, x_test, y_test

    except Exception as e:
        print(f"Error in function prepare_for_cnn: {e}")
        return None, None, None, None
