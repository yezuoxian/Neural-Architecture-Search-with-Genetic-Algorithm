from __future__ import division

import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        return 0.5 * np.sum((target - input) ** 2) / len(input)

    def backward(self, input, target):
        '''Your codes here'''
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        self.prob = (np.exp(input).T / np.exp(input).sum(axis=1)).T
        return - np.sum(target * np.log(self.prob)) / len(input)

    def backward(self, input, target):
        '''Your codes here'''
        return (self.prob - target) / len(input)
