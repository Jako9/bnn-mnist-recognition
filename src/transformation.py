import random

class ThresholdTransform(object):
    def __init__(self, max_val):
        self.max_val = max_val / \
            255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.max_val).to(x.dtype)  # do not change the data type

class ProbabilityTransform(object):
    def __init__(self, max_val):
        self.max_val = max_val / \
            255.

    def __call__(self, x):
        #print("x = " + str(x) + ", rand = " + str(random.uniform(0, self.max_val)))
        return (x > random.uniform(0, self.max_val)).to(x.dtype)  # do not change the data type
