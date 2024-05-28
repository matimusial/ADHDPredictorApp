import copy
from scipy import signal
import numpy as np

from EEG.config import CUTOFFS, FS, CNN_INPUT_SHAPE


def filter_eeg_data(ADHD_DATA, CONTROL_DATA=None, band_type=2):
    """
    Filters EEG data.

    Args:
    ADHD_DATA (list): List of EEG data for the ADHD group.
    CONTROL_DATA (list, optional): List of EEG data for the control group. Required for version 1.
    band_type (int): Type of band to filter. Default is 2.

    Returns:
    list: Filtered EEG data or tuple of two lists if CONTROL_DATA is provided.
    """
    order = 4
    cutoff = CUTOFFS[band_type]
    low_cutoff = cutoff[0]
    high_cutoff = cutoff[1]
    b, a = signal.butter(order, [low_cutoff / (0.5 * FS), high_cutoff / (0.5 * FS)], btype='bandpass')

    if CONTROL_DATA is not None:
        ADHD_FILTERED = []
        CONTROL_FILTERED = []

        for i in range(len(ADHD_DATA)):
            ADHD_FILTERED.append(signal.filtfilt(b, a, ADHD_DATA[i]))

        for i in range(len(CONTROL_DATA)):
            CONTROL_FILTERED.append(signal.filtfilt(b, a, CONTROL_DATA[i]))

        return ADHD_FILTERED, CONTROL_FILTERED

    else:
        ADHD_FILTERED = []
        for i in range(len(ADHD_DATA)):
            ADHD_FILTERED.append(signal.filtfilt(b, a, ADHD_DATA[i]))
        return ADHD_FILTERED


def normalize_eeg_data(ADHD_DATA, CONTROL_DATA=None):
    """
    Normalizes EEG data.

    Args:
    ADHD_DATA (list): List of EEG data for the ADHD group.
    CONTROL_DATA (list, optional): List of EEG data for the control group.

    Returns:
    list: Normalized EEG data or tuple of two lists if CONTROL_DATA is provided.
    """

    def normalize(data):
        data_normalized = copy.deepcopy(data)
        for i in range(len(data)):
            for j in range(CNN_INPUT_SHAPE[0]):
                min_value = np.min(data_normalized[i][j]).astype(np.float64)
                max_value = np.max(data_normalized[i][j]).astype(np.float64)
                if max_value != min_value:
                    data_normalized[i][j] = 2 * ((data_normalized[i][j] - min_value) / (max_value - min_value)) - 1
                else:
                    data_normalized[i][j] = np.zeros_like(data_normalized[i][j])
        return data_normalized

    if CONTROL_DATA is not None:
        if (len(ADHD_DATA) <= 1) or (len(CONTROL_DATA) <= 1):
            raise ValueError("Ta funkcja wymaga więcej niż jednego pacjenta w każdej grupie")

        ADHD_DATA_normalized = normalize(ADHD_DATA)
        CONTROL_DATA_normalized = normalize(CONTROL_DATA)

        return ADHD_DATA_normalized, CONTROL_DATA_normalized

    else:
        if len(ADHD_DATA) <= 1:
            raise ValueError("Ta funkcja wymaga więcej niż jednego pacjenta")

        ADHD_DATA_normalized = normalize(ADHD_DATA)
        return ADHD_DATA_normalized


def clip_eeg_data(ADHD_DATA, CONTROL_DATA=None):
    """
    Clips EEG data to a specified percentile.

    Args:
    ADHD_DATA (list): List of EEG data for the ADHD group.
    CONTROL_DATA (list, optional): List of EEG data for the control group.

    Returns:
    list: Clipped EEG data or tuple of two lists if CONTROL_DATA is provided.
    """
    percentile = 99.8

    def clip_data(data):
        data_clipped = copy.deepcopy(data)

        for i in range(len(data)):
            for j in range(CNN_INPUT_SHAPE[0]):
                channel_data = data[i][j]
                threshold = np.abs(np.percentile(channel_data, percentile))
                clipped_data = np.clip(channel_data, a_min=-threshold, a_max=threshold)
                data_clipped[i][j] = clipped_data

        return data_clipped

    if CONTROL_DATA is not None:
        if (len(ADHD_DATA) <= 1) or (len(CONTROL_DATA) <= 1):
            raise ValueError("Ta funkcja wymaga więcej niż jednego pacjenta w każdej grupie")

        ADHD_CLIPPED = clip_data(ADHD_DATA)
        CONTROL_CLIPPED = clip_data(CONTROL_DATA)

        return ADHD_CLIPPED, CONTROL_CLIPPED

    else:
        if len(ADHD_DATA) <= 1:
            raise ValueError("Ta funkcja wymaga więcej niż jednego pacjenta")

        ADHD_CLIPPED = clip_data(ADHD_DATA)
        return ADHD_CLIPPED
