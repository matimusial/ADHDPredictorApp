import numpy as np

from MRI.config import CNN_SINGLE_INPUT_SHAPE_MRI


def make_predict_data(adhd_raw, control_raw):
    """Prepares predict data.

    Args:
        adhd_raw (list or np.array): Raw ADHD data.
        control_raw (list or np.array): Raw control data.

    Returns:
        X_pred (np.array): Prediction data.
        y_pred (np.array): Labels for prediction data.
        adhd_updated (list): Updated ADHD data after removing prediction samples.
        control_updated (list): Updated control data after removing prediction samples.
    """
    adhd_pred = []
    adhd_updated = []
    control_pred = []
    control_updated = []

    adhd_random_indices = np.random.choice(range(len(adhd_raw)), size=5, replace=False)
    control_random_indices = np.random.choice(range(len(control_raw)), size=5, replace=False)

    for i in range(len(adhd_raw)):
        if i in adhd_random_indices:
            adhd_pred.append(adhd_raw[i])
        else:
            adhd_updated.append(adhd_raw[i])

    for i in range(len(control_raw)):
        if i in control_random_indices:
            control_pred.append(control_raw[i])
        else:
            control_updated.append(control_raw[i])

    y_adhd = np.ones(len(adhd_pred))
    y_control = np.zeros(len(control_pred))
    y_pred = np.hstack((y_adhd, y_control))

    X_adhd = np.reshape(adhd_pred, (len(adhd_pred), CNN_SINGLE_INPUT_SHAPE_MRI, CNN_SINGLE_INPUT_SHAPE_MRI, 1))
    X_control = np.reshape(control_pred, (len(control_pred), CNN_SINGLE_INPUT_SHAPE_MRI, CNN_SINGLE_INPUT_SHAPE_MRI, 1))

    X_pred = np.vstack((X_adhd, X_control))

    return X_pred, y_pred, adhd_updated, control_updated
