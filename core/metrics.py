import numpy as np


def calculate_entropy(y):
    # find unique classes
    classes, count = np.unique(y, return_counts=True)
    probabilities = count / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy


def calculate_gain_raio(y_parent, y_childern):
    parent_entropy = calculate_entropy(y_parent)

    n_parent = len(y_parent)
    weighted_child_entropy = 0
    split_info = 0

    for y_child in y_childern:
        n_child = len(y_child)
        weight = n_child / n_parent
        weighted_child_entropy += weight * calculate_entropy(y_child)
        # split info C4.5
        split_info -= weight * np.log2(weight + 1e-9)

    # calculate info gain
    info_gain = parent_entropy - weighted_child_entropy

    # avoid defision by 0
    if split_info == 0:
        return 0

    gain_ratio = info_gain / split_info

    return gain_ratio
