import numpy as np

from brainfeaturedecode.feature_generation.abstract_feature_generator import \
    AbstractFeatureGenerator
from brainfeaturedecode.utils.file_util import json_load


class MetaFeatureGenerator(AbstractFeatureGenerator):

    def get_feature_labels(self):
        """
        :return: list of feature labels of the form <domain>_<feature>
        """
        return self.get_feature_names()

    def get_feature_names(self):
        """
        :return: basically a list with shortened names from above in the form <domain>_<name>
        """
        return [self.domain + '_' + feat for feat in self.meta_feats]

    def generate_features(self, file_name):
        info = json_load(file_name.replace(file_name.split('.')[-1], "json"))
        age = info["age"]
        gender = 0 if info["gender"] == "M" else 1
        gender = gender
        meta_feats = np.array([age, gender])

        # repeat age and gender if returning features per epoch
        if not self.agg_mode:
            meta_feats = np.repeat(meta_feats, self.n_epochs)
            meta_feats = meta_feats.reshape(self.n_epochs, -1)

        return meta_feats

    def __init__(self, elecs, agg, domain="meta", n_epochs=None):
        super(MetaFeatureGenerator, self).__init__(
            domain=domain, electrodes=elecs, agg_mode=agg)
        self.meta_feats = ["age", "gender"]
        self.n_epochs = n_epochs
