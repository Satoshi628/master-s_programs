import numpy as np


def otsu_threshold(tensor, bins=100):
    _max = tensor.max()
    _min = tensor.min()
    delta = (_max - _min)/bins
    
    steps = np.arange(_min, _max, delta)
    
    threshold = 0
    sigmma_b = 0.
    for idx, th in enumerate(steps):
        class0 = tensor < th
        average0 = tensor[class0].mean()
        average1 = tensor[~class0].mean()
        _sigmma = class0.sum()*(~class0).sum()*(average0 - average1)**2
        if sigmma_b < _sigmma:
            threshold = th
            sigmma_b = _sigmma

    return threshold
