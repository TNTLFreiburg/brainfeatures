import numpy as np

from brainfeatures.feature_generation.abstract_feature_generator import (
    AbstractFeatureGenerator)
from brainfeatures.feature_generation import features_frequency


class FrequencyFeatureGenerator(AbstractFeatureGenerator):
    """ computes features in the frequency domain implemented in features_
    frequency using fourier transform """
    def get_feature_labels(self):
        """
        :return: list of feature labels of the form
        <fft>_<feature>_<lower-upperHz>_<channel>
        """
        feature_labels = []
        for freq_feat in self.freq_feats:
            freq_feat = freq_feat.replace("_", "-")
            for band_id, band in enumerate(self.bands):
                lower, upper = band
                for electrode in self.electrodes:
                    label = '_'.join([
                        self.domain,
                        freq_feat,
                        str(lower) + '-' + str(upper) + 'Hz',
                        str(electrode)])
                    feature_labels.append(label)
        return feature_labels

    def convert_with_fft(self, weighted_epochs):
        epochs_amplitudes = np.abs(np.fft.rfft(weighted_epochs, axis=2))
        epochs_amplitudes /= weighted_epochs.shape[-1]
        return epochs_amplitudes
    
    def generate_features(self, weighted_epochs):
        """ computes all frequency domain features as implemented in module
        features_frequency
        :param weighted_epochs: ndarray with split eeg data weighted by a
            window function in shape of n_eochs x n_elecs x n_samples_in_epoch
        :return: ndarray of features in shape [n_epochs x] n_elecs x n_bands x
            n_freq_features
        """
        (n_epochs, n_elecs, n_samples_in_epoch) = weighted_epochs.shape
        epochs_psds = self.convert_with_fft(weighted_epochs)
        freq_bin_size = self.sfreq / n_samples_in_epoch
        freqs = np.fft.fftfreq(int(n_samples_in_epoch), 1. / self.sfreq)

        # extract frequency bands and generate features
        # n_epochs x n_elecs x n_bands x n_feats
        freq_feats = np.ndarray(shape=(n_epochs, len(self.freq_feats),
                                       len(self.bands), n_elecs))
        for freq_feat_id, freq_feat_name in enumerate(self.freq_feats):
            # assumes that "power" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            if freq_feat_name == "power_ratio":
                powers = freq_feats[:, self.freq_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                func = getattr(features_frequency, freq_feat_name)
                ratio = func(powers, axis=-2)
                freq_feats[:, freq_feat_id, :, :] = ratio
            # assumes that "ratio" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            elif freq_feat_name == "spectral_entropy":
                func = getattr(features_frequency, freq_feat_name)
                ratios = freq_feats[:, self.freq_feats.index("power_ratio"),:,:]
                spec_entropy = func(ratios)
                freq_feats[:, freq_feat_id, :, :] = spec_entropy
            else:
                func = getattr(features_frequency, freq_feat_name)
                # amplitudes shape: epochs x electrodes x frequencies
                band_psd_features = np.ndarray(shape=(n_epochs, len(self.bands),
                                                      n_elecs))
                for band_id, (lower, upper) in enumerate(self.bands):
                    lower_bin, upper_bin = (int(lower / freq_bin_size),
                                            int(upper / freq_bin_size))
                    # if upper_bin corresponds to nyquist frequency or higher,
                    # take last available frequency
                    if upper_bin >= len(freqs):
                        upper_bin = len(freqs) - 1
                    band_psds = np.take(epochs_psds,
                                        range(lower_bin, upper_bin), axis=-1)
                    band_psd_features[:, band_id, :] = func(band_psds, axis=-1)

                freq_feats[:, freq_feat_id, :, :] = band_psd_features

        freq_feats = freq_feats.reshape(n_epochs, -1)
        # aggregate over the dimension of epochs
        if self.agg_mode:
            freq_feats = self.agg_mode(freq_feats, axis=0)

        return freq_feats

    def __init__(self, elecs, agg, bands, sfreq, domain="fft"):
        super(FrequencyFeatureGenerator, self).__init__(
            domain=domain, electrodes=elecs, agg_mode=agg)
        self.freq_feats = sorted([
            feat_func
            for feat_func in dir(features_frequency)
            if not feat_func.startswith('_')])
        self.bands = bands
        self.sfreq = sfreq
