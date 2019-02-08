import numpy as np

from brainfeatures.feature_generation.abstract_feature_generator import \
    AbstractFeatureGenerator
from brainfeatures.feature_generation import features_time as features_time


class TimeFeatureGenerator(AbstractFeatureGenerator):
    """ computes features in the time domain implemented in features_time """

    def get_feature_labels(self):
        """
        :return: list of feature labels of the form <domain>_<feature>_<channel>
        """
        feature_labels = []
        for time_feat in self.time_feats:
            time_feat = time_feat.replace("_", "-")
            for electrode in self.electrodes:
                label = '_'.join([self.domain, time_feat, str(electrode)])
                feature_labels.append(label)
        return feature_labels

    def generate_features(self, windows):
        """ computes all time domain features specified by self.time_feats and
        implemented in features_time.py
        :param windows: ndarray with split eeg data in shape of
            n_windows x n_elecs x n_samples_in_window
        :return: ndarray with eeg time features in shape of
            n_windows x n_elecs x n_time_features
        """
        (n_windows, n_elecs, n_samples_in_window) = windows.shape
        time_feats = np.ndarray(
            shape=(n_windows, len(self.time_feats), n_elecs))
        for time_feat_id, time_feat_name in enumerate(self.time_feats):
            func = getattr(features_time, time_feat_name)
            time_feats[:, time_feat_id, :] = func(
                windows, -1, Kmax=self.Kmax, n=self.n, T=self.T, Tau=self.Tau,
                DE=self.DE, W=self.W, fs=self.sfreq)

        time_feats = time_feats.reshape(n_windows, -1)
        # aggregate over the dimension of epochs
        if self.agg_mode:
            time_feats = self.agg_mode(time_feats, axis=0)
        return time_feats

    def __init__(self, elecs, agg, sfreq, outlier_value, domain="time"):
        super(TimeFeatureGenerator, self).__init__(
            domain=domain, electrodes=elecs, agg_mode=agg)
        self.time_feats = sorted([
            feat_func
            for feat_func in dir(features_time)
            if not feat_func.startswith('_')])

        self.outlier_value = outlier_value

        # for computation of pyeeg features
        self.Kmax = 3
        self.n = 4
        self.T = 1
        self.Tau = 4
        self.DE = 10
        self.W = None
        self.sfreq = sfreq