import smo
from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractSVM(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_coef(self):
        pass


class LinearSVM(AbstractSVM):
    dec_fun = np.vectorize(lambda x: -1 if x < 0 else 1)

    def __init__(self, c=1.0, eps=1e-4, iterations=1000):
        self.C = c
        self.eps = eps
        self.iter = iterations
        self.bias: float = None
        self.weights: np.ndarray = None

    @staticmethod
    def add_bias(x: np.ndarray):
        m, n = x.shape
        bias = np.ones((m, 1))
        return np.hstack((bias, x))
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        x, y = x.astype(float), y.astype(int)
        x = LinearSVM.add_bias(x)
        self.weights, self.bias = smo.smo(x, y, self.iter, self.C, self.eps)

    def predict(self, x: np.ndarray):
        x = x.astype(float)
        val = np.add(x.dot(self.weights), self.bias)
        return LinearSVM.dec_fun(val)

    def get_coef(self):
        return self.bias, self.weights
    
    def score(self, x: np.ndarray, y: np.ndarray):
        x, y = x.astype(float), y.astype(int)
        val = self.predict(x)
        return np.count_nonzero(y == val) / x.shape[0]
