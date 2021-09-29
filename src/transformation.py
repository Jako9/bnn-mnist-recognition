import random
import torch
import numpy as np
import time
#pip install numba
from numba import njit

class ThresholdTransform(object):
    """Binarize Tensor via a threshold
    """
    def __init__(self, max_val):
        """Initialize tranformer

        Args:
            max_val (int): threshold for transformation
        """
        self.max_val = max_val / \
            255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        """Binarization of Tensor

        Args:
            x (torch.Tensor): image in tensor-form

        Returns:
            torch.Tensor: Same image in tensor-form but binarized
        """
        return (x > self.max_val).to(x.dtype)  # do not change the data type
s
class ProbabilityTransform(object):
    """Binarize Tensor via probability
    """
    def __call__(self, x):
        """Binarization of Tensor

        Args:
            x (torch.Tensor): image in tensor-form

        Returns:
            torch.Tensor: Same image in tensor-form but binarized
        """
        binarizedPicture = x.numpy()
        return torch.Tensor(fastBinarization(binarizedPicture))

#Njit optimized iteration through pixelarray
@njit()
def fastBinarization(binarizedPicture):
    """Loop through array and replace floats with bits

    Args:
        binarizedPicture (numpy.array): image in array-form

    Returns:
        numpy.array: Binarized version of array
    """
    for i,dim1 in enumerate(binarizedPicture[0]):
        for j,dim2 in enumerate(dim1):
            binarizedPicture[0,i,j] = dim2 > random.uniform(0, 1)
    return binarizedPicture
