from abc import ABC, abstractmethod


class AbstractFeatureGenerator(ABC):
    def __init__(self, domain, electrodes, agg_mode):
        self.domain = domain
        self.electrodes = electrodes
        self.agg_mode = agg_mode
        self.times_list = []

    @abstractmethod
    def generate_features(self, data):
        raise NotImplementedError

    @abstractmethod
    def get_feature_labels(self):
        raise NotImplementedError
