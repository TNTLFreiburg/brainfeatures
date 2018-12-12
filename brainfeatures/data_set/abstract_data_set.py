from abc import ABCMeta, abstractmethod

"""
custom data set can be inherited from DataSet. it has to implement a 
__getitem__ and a __len__ function if data set should be used on raw data, 
__getitem__ should return a 3-tuple of signals array, sampling frequency and 
label if data set should be used on cleaned data or features, __getitem__ 
should return a 2-tuple of signals array and label
"""


class DataSet(object):
    """
    Abstract class of a data set specifying functions and members that are
    needed by the experiment class.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        To override. Has to return (example, sampling frequency, label) of
        the given index
        """
        example = [[1, 2, 3], [3, 4, 5]]
        sfreq = 100
        label = True
        return example, sfreq, label

    @abstractmethod
    def __len__(self):
        """
        To override. Has to return number of examples
        """
        length = 1
        return length
