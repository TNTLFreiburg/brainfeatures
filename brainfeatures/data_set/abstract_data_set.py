from abc import ABCMeta, abstractmethod


class DataSet(object):
    """ custom data set can be inherited from DataSet. it has to implement a
    __getitem__ and a __len__ function returning a 3-tuple of signals array,
    sampling frequency and label and giving the number of examples in the
    data set, respectively
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
