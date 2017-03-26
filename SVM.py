import smo
from abc import ABCMeta, abstractmethod


class AbstractSVM(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x, y):
        pass

    @abstractmethod
    def get_coef(self):
        pass


class LinearSVM(AbstractSVM):
    def __init__(self):