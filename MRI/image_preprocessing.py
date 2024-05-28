import numpy as np
import copy


def normalize(data):
    """
    Normalizes list of images to the range [-1, 1].
    """
    normalized = np.empty_like(data)
    for i in range(len(data)):
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        normalized[i] = (data[i] - min_val) / (max_val - min_val)
        normalized[i] = 2 * normalized[i] - 1
    return normalized


def check_dimensions(data):
    """
    Checks list of images if the dimensions of the data are square.
    """
    for i, item in enumerate(data):
        rows, columns = item.shape
        if rows != columns:
            print(f"Data {i} ma wymiary nie będące kwadratem: {rows, columns}")


def trim_rows(data, nr_rows=4):
    """
    Trims the list of images by removing a specified number of rows from each side.
    """
    trimmed = copy.deepcopy(data)
    for i in range(len(data)):
        trimmed[i] = data[i][nr_rows:-nr_rows]
    return trimmed


def trim_one(data, nr_rows=4):
    """
    Trims the single image by removing a specified number of rows from each side.
    """
    return data[nr_rows:-nr_rows, nr_rows:-nr_rows]
