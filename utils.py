import numpy as np

def get_tensor_from_2D_dataset(dataset):
    """From a tensorflow dataset, return a list of tensor.
    """

    Xs = []
    ys = []
    for x, y in dataset:
        Xs.append(x)
        ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

def get_tensor_from_3D_dataset(dataset):
    Xs = []
    masks = []
    labels = []
    for _x, _mask, label in dataset:

        Xs.append(_x)
        masks.append(_mask)
        labels.append(label)

    X = np.concatenate(Xs, axis=0)
    
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)

    return X, masks, labels