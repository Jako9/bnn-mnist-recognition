import random
import torch
import numpy as np

class ThresholdTransform(object):
    def __init__(self, max_val):
        self.max_val = max_val / \
            255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.max_val).to(x.dtype)  # do not change the data type

class ProbabilityTransform(object):

    def __call__(self, x):
        binarizedPicture = x.numpy()
        for i,dim1 in enumerate(binarizedPicture[0]):
            for j,dim2 in enumerate(dim1):
                binarizedPicture[0,i,j]= dim2 > random.uniform(0, 1)

        #print(binarizedPicture)
        return torch.Tensor(binarizedPicture)
