from pywt._extensions._pywt import (DiscreteContinuousWavelet,
    ContinuousWavelet, Wavelet, _check_dtype)
from pywt._functions import (integrate_wavelet, scale2frequency,
    central_frequency)
from pywt import wavedec, dwt_max_level, wavelist
import numpy as np

from brainfeatures.feature_generation.abstract_feature_generator import (
    AbstractFeatureGenerator)
from brainfeatures.feature_generation import features_wavelets


class WaveletFeatureGenerator(AbstractFeatureGenerator):
    """ computes features in the time-frequency domain implemented in
    features_wavelets using wavelet transforms """

    def get_feature_labels(self):
        """
        :return: list of feature labels of the form <domain>_<feature>_<channel>
        """
        feature_labels = []
        for wt_feat in self.wt_feats:
            for level in self.levels:
                for electrode in self.electrodes:
                    feature_labels.append(
                        '_'.join([self.domain, wt_feat, level, str(electrode)]))
        return feature_labels

    def freqs_to_scale(self, freqs, wavelet, sfreq):
        """ compute cwt scales to given frequencies """
        return [self.freq_to_scale(freq[1], wavelet, sfreq) for freq in freqs]

    def freq_to_scale(self, freq, wavelet, sfreq):
        """ compute cwt scale to given frequency
        see: https://de.mathworks.com/help/wavelet/ref/scal2frq.html """
        central_freq = central_frequency(wavelet)
        assert freq > 0, "freq smaller or equal to zero!"
        scale = central_freq / freq
        return scale * sfreq

    def generate_level_names(self, n_levels):
        """ name the levels of the wavelet transformation, i.e. a1, d1, d0 """
        level_names = ["a" + str(n_levels-1)]
        for i in range(n_levels)[:-1][::-1]:
            level_names.append("d" + str(i))
        return level_names

    # taken from pywt and adapted to not compute and return frequencies
    # this is avialable in new, separate function
    # like this, it can be applied using numpy.apply_along_axis
    def pywt_cwt(self, data, scales, wavelet):
        """
        cwt(data, scales, wavelet)
        One dimensional Continuous Wavelet Transform.
        Parameters
        ----------
        data : array_like
            Input signal
        scales : array_like
            scales to use
        wavelet : Wavelet object or name
            Wavelet to use
        Returns
        -------
        coefs : array_like
            Continous wavelet transform of the input signal for the given
            scales and wavelet
        Notes
        -----
        Size of coefficients arrays depends on the length of the input array
        and the length of given scales.
        Examples
        --------
        """
        # accept array_like input; make a copy to ensure a contiguous array
        dt = _check_dtype(data)
        data = np.array(data, dtype=dt)
        if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
            wavelet = DiscreteContinuousWavelet(wavelet)
        if np.isscalar(scales):
            scales = np.array([scales])
        if data.ndim == 1:
            if wavelet.complex_cwt:
                out = np.zeros((np.size(scales), data.size), dtype=complex)
            else:
                out = np.zeros((np.size(scales), data.size))
            for i in np.arange(np.size(scales)):
                precision = 10
                int_psi, x = integrate_wavelet(wavelet, precision=precision)
                step = x[1] - x[0]
                j = np.floor(np.arange(
                    scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
                if np.max(j) >= np.size(int_psi):
                    j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
                coef = - np.sqrt(scales[i]) * np.diff(
                    np.convolve(data, int_psi[j.astype(np.int)][::-1]))
                d = (coef.size - data.size) / 2.
                out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
            #        frequencies = scale2frequency(wavelet, scales, precision)
            #        if np.isscalar(frequencies):
            #            frequencies = np.array([frequencies])
            #        for i in np.arange(len(frequencies)):
            #            frequencies[i] /= sampling_period
            #        return (out, frequencies)
            return out
        else:
            raise ValueError("Only dim == 1 supportet")

    def pywt_freqs(self, wavelet, scales, sampling_period=1.):
        """ taken from pywt.cwt and split into independent function. if
        frequencies are desired, returns them wrt
         wavelet, scales and sampling period of the signal """
        if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
            wavelet = DiscreteContinuousWavelet(wavelet)
        precision = 10
        frequencies = scale2frequency(wavelet, scales, precision)
        if np.isscalar(frequencies):
            frequencies = np.array([frequencies])
        for i in np.arange(len(frequencies)):
            frequencies[i] /= sampling_period
        return frequencies

    def generate_cwt_features(self, weighted_windows):
        """
        :param weighted_windows: ndarray with split eeg data weighted by a
            window function in shape of n_windows x n_elecs x
            n_samples_in_window
        :return: ndarray of features in shape n_windows x n_elecs x
        n_features x n_bands
        """
        (n_windows, n_elecs, n_samples_in_window) = weighted_windows.shape
        scales = self.freqs_to_scale(self.band_limits, self.wavelet,
                                     self.sfreq)
        if self.levels is None:
            self.levels = self.generate_level_names(len(scales))
        coefficients = np.apply_along_axis(
            func1d=self.pywt_cwt, axis=2, arr=weighted_windows, scales=scales,
            wavelet=self.wavelet)
        # n_windows x n_elecs x n_levels x n_coefficients
        coefficients = np.swapaxes(coefficients, 1, 2)
        coefficients = np.abs(coefficients) / weighted_windows.shape[-1]

        cwt_feats = np.ndarray(shape=(n_windows, len(self.wt_feats),
                                      len(scales), n_elecs))
        for wt_feat_id, wt_feat_name in enumerate(self.wt_feats):
            if wt_feat_name == "power_ratio":
                func = getattr(features_wavelets, wt_feat_name)
                powers = cwt_feats[:, self.wt_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                feats = func(powers, axis=-1)
            elif wt_feat_name == "spectral_entropy":
                ratios = cwt_feats[:, self.wt_feats.index("power_ratio"), :, :]
                func = getattr(features_wavelets, wt_feat_name)
                feats = func(ratios)
            else:
                func = getattr(features_wavelets, wt_feat_name)
                feats = func(coefficients=coefficients, axis=-1)
            cwt_feats[:, wt_feat_id, :, :] = feats

        cwt_feats = cwt_feats.reshape(n_windows, -1)
        if self.agg_mode:
            cwt_feats = self.agg_mode(cwt_feats, axis=0)

        return cwt_feats

    def generate_dwt_features(self, weighted_windows):
        """ computes all wt domain features
        :param weighted_windows: ndarray with split eeg data weighted by a
        window function in shape of n_windows x n_elecs x n_samples_in_window
        :return: ndarray of features in shape n_windows x n_elecs x n_features
        x n_bands
        """
        (n_windows, n_elecs, n_samples_in_window) = weighted_windows.shape
        if self.levels is None:
            max_level = dwt_max_level(n_samples_in_window, self.wavelet)
            pseudo_freqs = [self.sfreq/2**i for i in range(1, max_level)]
            pseudo_freqs = [pseudo_freq for pseudo_freq in pseudo_freqs
                            if pseudo_freq >= 2]
            self.levels = self.generate_level_names(len(pseudo_freqs))
        n_levels = len(self.levels)
        dwt_feats = np.ndarray(
            shape=(n_windows, len(self.wt_feats), n_levels, n_elecs)
        )
        # list of length n_bands of ndarray: x n_epochs x n_channels x
        # n_band_coeffs
        multi_level_coeffs = wavedec(data=weighted_windows,
                                     wavelet=self.wavelet, level=n_levels-1,
                                     axis=2)
        multi_level_coeffs = [np.abs(d) for d in multi_level_coeffs]
        multi_level_coeffs = [d/weighted_windows.shape[-1] for d in
                              multi_level_coeffs]

        for wt_feat_id, wt_feat_name in enumerate(self.wt_feats):
            # assumes that "power" feature was already computed. which should
            # be the case, since features are iterated alphabetically
            if wt_feat_name == "power_ratio":
                func = getattr(features_wavelets, wt_feat_name)
                powers = dwt_feats[:, self.wt_feats.index("power"), :, :]
                # divide the power by the sum of powers in each band to gain
                # power ratio feature
                ratios = func(powers)
                dwt_feats[:, wt_feat_id, :, :] = ratios
            elif wt_feat_name == "spectral_entropy":
                func = getattr(features_wavelets, wt_feat_name)
                ratios = dwt_feats[:, self.wt_feats.index("power"), :, :]
                spec_entropy = func(ratios)
                dwt_feats[:, wt_feat_id, :, :] = spec_entropy
            else:
                func = getattr(features_wavelets, wt_feat_name)
                # use apply_along_axis here?
                for level_id, level_coeffs in enumerate(multi_level_coeffs):
                    level_coeffs = np.abs(level_coeffs)
                    level_feats = func(coefficients=level_coeffs, axis=2)
                    dwt_feats[:, wt_feat_id, level_id, :] = level_feats

        dwt_feats = dwt_feats.reshape(n_windows, -1)
        if self.agg_mode:
            dwt_feats = self.agg_mode(dwt_feats, axis=0)

        # n_elecs x n_levels * n_feats
        return dwt_feats

    def generate_features(self, weighted_windows):
        """ generate either cwt or dwt features using pywavelets """
        if self.domain == "cwt":
            features = self.generate_cwt_features(weighted_windows)
        else:
            assert self.domain == "dwt"
            features = self.generate_dwt_features(weighted_windows)
        return features

    def __init__(self, domain, elecs, agg, sfreq, wavelet, band_limits):
        super(WaveletFeatureGenerator, self).__init__(
            domain=domain, electrodes=elecs, agg_mode=agg)
        self.wt_feats = sorted([
            feat_func
            for feat_func in dir(features_wavelets)
            if not feat_func.startswith('_')])
        assert wavelet in wavelist(), "unknown wavelet {}".format(wavelet)
        if wavelet in wavelist(kind="discrete"):
            self.wavelet = Wavelet(wavelet)
        else:
            self.wavelet = ContinuousWavelet(wavelet)
        self.sfreq = sfreq
        self.band_limits = band_limits
        self.levels = None
