from pywt._extensions._pywt import DiscreteContinuousWavelet, \
    ContinuousWavelet, Wavelet, _check_dtype
from pywt._functions import integrate_wavelet, scale2frequency, \
    central_frequency
from pywt import wavedec, dwt_max_level, wavelist

import pandas as pd
import numpy as np
import logging
import os

from ..utils.file_util import json_load
from ..utils.data_util import split_into_epochs, \
    apply_window_function, filter_to_frequency_bands, \
    reject_windows_with_outliers, assemble_overlapping_band_limits
from . import features_time as ft
from . import features_wavelets as fw
from . import features_phase as fp
from . import features_frequency as ff


class WaveletTransformer(object):
    def __init__(self):
        pass

    def freq_to_scale(self, freq, wavelet, sfreq):
        """ compute cwt scale to given frequency
        see: https://de.mathworks.com/help/wavelet/ref/scal2frq.html """
        central_freq = central_frequency(wavelet)
        assert freq > 0, "freq smaller or equal to zero!"
        scale = central_freq / freq
        return scale * sfreq

    def freqs_to_scale(self, freqs, wavelet, sfreq):
        """ compute cwt scales to given frequencies """
        scales = []
        for freq in freqs:
            scale = self.freq_to_scale(freq[1], wavelet, sfreq)
            scales.append(scale)
        return scales

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

    def cwt_transform(self, crops, wavelet, band_limits, sfreq):
        scales = self.freqs_to_scale(
            freqs=band_limits, wavelet=wavelet, sfreq=sfreq)
        coefficients = np.apply_along_axis(
            func1d=self.pywt_cwt, axis=2, arr=crops, scales=scales,
            wavelet=wavelet)
        # n_windows x n_elecs x n_levels x n_coefficients
        coefficients = np.swapaxes(a=coefficients, axis1=1, axis2=2)
        # coefficients = np.abs(coefficients)
        return coefficients

    def dwt_transform(self, crops, wavelet, sfreq):
        (n_windows, n_elecs, n_samples_in_window) = crops.shape
        max_level = dwt_max_level(
            data_len=n_samples_in_window, filter_len=wavelet)
        pseudo_freqs = [sfreq/2**i for i in range(1, max_level)]
        # don't take pseudo freqs < 2
        pseudo_freqs = [pseudo_freq for pseudo_freq in pseudo_freqs
                        if pseudo_freq >= 2]
        n_levels = len(pseudo_freqs)
        # list of length n_bands of ndarray: x n_epochs x n_channels x
        # n_band_coeffs
        multi_level_coeffs = wavedec(
            data=crops, wavelet=wavelet, level=n_levels-1, axis=2)
        # multi_level_coeffs = np.abs(multi_level_coeffs)
        return multi_level_coeffs


class FourierTransformer(object):
    def __init__(self):
        pass

    def convert_with_fft(self, crops):
        epochs_amplitudes = np.abs(np.fft.rfft(crops, axis=2))
        epochs_amplitudes /= crops.shape[-1]
        return epochs_amplitudes

    def dft_transform(self, crops, sfreq, n_samples_in_epoch):
        crop_psds = self.convert_with_fft(crops=crops)
        freq_bin_size = sfreq / n_samples_in_epoch
        freqs = np.fft.fftfreq(int(n_samples_in_epoch), 1. / sfreq)
        return crop_psds, freqs, freq_bin_size


def generate_cwt_features(crops, wavelet, band_limits, agg_func, sfreq):
    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape
    wt = WaveletTransformer()
    cwt_coefficients = wt.cwt_transform(
        crops=crops, wavelet=wavelet,
        band_limits=band_limits, sfreq=sfreq)

    cwt_feats = np.ndarray(shape=(n_crops, 7, len(cwt_coefficients), n_elecs))
    cwt_feats[:, 0, :, :] = fw.bounded_variation(
            coefficients=cwt_coefficients, axis=2)
    cwt_feats[:, 1, :, :] = fw.entropy(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 2, :, :] = fw.maximum(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 3, :, :] = fw.mean(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 4, :, :] = fw.minimum(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 5, :, :] = fw.power(coefficients=cwt_coefficients, axis=3)
    cwt_feats[:, 6, :, :] = fw.variance(coefficients=cwt_coefficients, axis=3)

    powers = cwt_feats[:, 5, :, :]
    ratios = powers / np.sum(powers, axis=1, keepdims=True)
    cwt_feats[:, 7, :, :] = ratios

    cwt_feats = cwt_feats.reshape(n_crops, -1)
    if agg_func is not None:
        cwt_feats = agg_func(cwt_feats, axis=0)

    return cwt_feats


def generate_dwt_features(crops, wavelet, agg_func, band_limits, sfreq):
    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape
    wt = WaveletTransformer()
    dwt_coefficients = wt.dwt_transform(
        crops=crops, wavelet=wavelet, sfreq=sfreq)

    dwt_feats = np.ndarray(
        shape=(n_crops, 7, len(dwt_coefficients), n_elecs))

    for level_id, level_coeffs in enumerate(dwt_coefficients):
        level_coeffs = np.abs(level_coeffs)
        dwt_feats[:, 0, level_id, :] = fw.bounded_variation(
            coefficients=level_coeffs, axis=2)
        dwt_feats[:, 1, level_id, :] = fw.entropy(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 2, level_id, :] = fw.maximum(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 3, level_id, :] = fw.mean(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 4, level_id, :] = fw.minimum(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 5, level_id, :] = fw.power(coefficients=level_coeffs, axis=2)
        dwt_feats[:, 6, level_id, :] = fw.variance(coefficients=level_coeffs, axis=2)

    powers = dwt_feats[:, 5, :, :]
    ratios = powers / np.sum(powers, axis=1, keepdims=True)
    dwt_feats[:, 7, :, :] = ratios

    dwt_feats = dwt_feats.reshape(n_crops, -1)
    if agg_func is not None:
        dwt_feats = agg_func(dwt_feats, axis=0)
    return dwt_feats


def generate_dft_features(crops, sfreq, band_limits, agg_func):
    (n_crops, n_elecs, n_samples_in_epoch) = crops.shape
    ftr = FourierTransformer()
    crops_amplitude_spectrum, freqs, freq_bin_size = ftr.dft_transform(
        crops=crops, sfreq=sfreq, n_samples_in_epoch=n_samples_in_epoch)

    # n_crops x n_freq_feats x n_bands x n_elecs
    freq_feats = np.ndarray(shape=(len(crops), 7 + 2, len(band_limits), n_elecs))
    for band_id, (lower, upper) in enumerate(band_limits):
        lower_bin, upper_bin = int(lower / freq_bin_size), int(upper / freq_bin_size)
        # if upper_bin corresponds to nyquist frequency or higher, take last available frequency
        if upper_bin >= len(freqs):
            upper_bin = len(freqs) - 1
        band_amplitude_spectrum = np.take(crops_amplitude_spectrum, range(lower_bin, upper_bin), axis=-1)

        freq_feats[:, 0, band_id, :] = ff.maximum(amplitude_spectrum=band_amplitude_spectrum)
        freq_feats[:, 1, band_id, :] = ff.mean(amplitude_spectrum=band_amplitude_spectrum)
        freq_feats[:, 2, band_id, :] = ff.minimum(amplitude_spectrum=band_amplitude_spectrum)
        freq_feats[:, 3, band_id, :] = ff.power(amplitude_spectrum=band_amplitude_spectrum)
        freq_feats[:, 4, band_id, :] = ff.value_range(amplitude_spectrum=band_amplitude_spectrum)
        freq_feats[:, 5, band_id, :] = ff.variance(amplitude_spectrum=band_amplitude_spectrum)
        freq_feats[:, 6, band_id, :] = freqs[lower_bin + ff.peak_frequency(amplitude_spectrum=band_amplitude_spectrum)]

    powers = freq_feats[:, 3, :, :]
    ratios = powers / np.sum(powers, axis=1, keepdims=True)
    freq_feats[:, 7, :, :] = ratios

    # TODO: check this
    spectral_entropy = np.sum([ratio * np.log(ratio) for ratio in ratios], axis=-1)
    spectral_entropy = -1 * spectral_entropy / np.log(ratios.shape[1])
    print(np.sum(spectral_entropy, axis=-1))
    freq_feats[:, 8, :, :] = spectral_entropy

    freq_feats = freq_feats.reshape(n_crops, -1)
    if agg_func is not None:
        freq_feats = agg_func(freq_feats, axis=0)
    return freq_feats


def generate_meta_features(file_name, agg_func, n_crops):
    json_file = file_name.replace(file_name.split('.')[-1], "json")
    if not os.path.exists(json_file):
        return None
    info = json_load(json_file)
    age = info["age"]
    gender = 0 if info["gender"] == "M" else 1
    meta_feats = np.array([age, gender])

    # repeat age and gender if returning features per epoch
    if agg_func is None:
        meta_feats = np.repeat(meta_feats, n_crops)
        meta_feats = meta_feats.reshape(n_crops, -1)
    return meta_feats


def generate_phase_features(signals, band_limits, sfreq, epoch_duration_s,
                            outlier_mask, agg_func):
    band_signals = filter_to_frequency_bands(
        signals=signals, bands=band_limits, sfreq=sfreq)
    band_crops = split_into_epochs(band_signals, sfreq=sfreq,
                                   epoch_duration_s=epoch_duration_s)
    band_crops = band_crops[outlier_mask == False]
    # n_windows x n_bands x n_elecs x n_samples_in_window
    epochs_instantaneous_phases = fp.instantaneous_phases(
        band_signals=band_crops, axis=-1)

    # n_windows x n_bands * n_signals*(n_signals-1)/2
    phase_locking_values = fp.phase_locking_values(
        inst_phases=epochs_instantaneous_phases)

    if agg_func is not None:
        # n_bands * n_signals*(n_signals-1)/2
        phase_locking_values = agg_func(phase_locking_values, axis=0)
    return phase_locking_values


def generate_time_features(crops, sfreq, agg_func, axis=-1):
    Kmax = 3
    n = 4
    T = 1
    Tau = 4
    DE = 10
    W = None

    dfa = ft.detrended_fluctuation_analysis(crops, axis=axis)
    energy_ = ft.energy(epochs=crops, axis=axis)
    fisher_info = ft.fisher_information(epochs=crops, axis=axis, Tau=Tau, DE=DE)
    fractal_dim = ft.fractal_dimension(epochs=crops, axis=axis)
    higuchi_fractal_dim = ft.higuchi_fractal_dimension(epochs=crops, axis=axis, Kmax=Kmax)
    [activity, mobility, complexity] = ft._hjorth_parameters(epochs=crops, axis=axis)
    hurst_exp = ft.hurst_exponent(epochs=crops, axis=axis)
    kurt = ft.kurtosis(epochs=crops, axis=axis)
    linelength = ft.line_length(epochs=crops, axis=axis)
    lyapunov_exp = ft.largest_lyauponov_exponent(epochs=crops, axis=axis, Tau=Tau, n=n, T=T, fs=sfreq)
    max_ = ft.maximum(epochs=crops, axis=axis)
    mean_ = ft.mean(epochs=crops, axis=axis)
    median_ = ft.median(epochs=crops, axis=axis)
    min_ = ft.minimum(epochs=crops, axis=axis)
    non_lin_energy = ft.non_linear_energy(epochs=crops, axis=axis)
    petrosian_fractal_dim = ft.petrosian_fractal_dimension(epochs=crops, axis=axis)
    skew = ft.skewness(epochs=crops, axis=axis)
    svd_entropy_ = ft.svd_entropy(epochs=crops, axis=axis, Tau=Tau, DE=DE, W=W)
    zero_crossings = ft.zero_crossing(epochs=crops, axis=axis)
    zero_crossings_dev = ft.zero_crossing_derivative(epochs=crops, axis=axis)

    time_features = np.hstack((
        dfa, energy_, fisher_info, fractal_dim, higuchi_fractal_dim, activity,
        complexity, mobility, hurst_exp, kurt, linelength, lyapunov_exp, max_,
        mean_, median_, min_, non_lin_energy, petrosian_fractal_dim, skew,
        svd_entropy_, zero_crossings, zero_crossings_dev))

    time_features = time_features.reshape(len(crops), -1)
    if agg_func is not None:
        time_features = agg_func(time_features, axis=0)
    return time_features


def generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, channels, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, file_name=None):

    non_overlapping_bands = band_limits
    if band_overlap:
        band_limits = assemble_overlapping_band_limits(band_limits)
    # split into epochs
    crops = split_into_epochs(
        signals=signals, sfreq=sfreq, epoch_duration_s=epoch_duration_s)
    # reject windows with outliers
    outlier_mask = reject_windows_with_outliers(
        outlier_value=max_abs_val, epochs=crops)
    crops = crops[outlier_mask == False]
    # inform and return if all epochs were removed
    if crops.size == 0:
        logging.warning("removed all crops due to outliers")
        return None, None
    # weight the samples by a window function
    n_samples_in_epoch = int(sfreq * epoch_duration_s)
    weighted_crops = apply_window_function(
        epochs=crops, window_name=window_name)

    cwt_features = generate_cwt_features(
        crops=weighted_crops, wavelet=continuous_wavelet, sfreq=sfreq,
        band_limits=non_overlapping_bands, agg_func=agg_mode)

    dwt_features = generate_dwt_features(
        crops=weighted_crops, wavelet=discrete_wavelet, sfreq=sfreq,
        agg_func=agg_mode, band_limits=non_overlapping_bands)

    dft_features = generate_dft_features(
        crops=weighted_crops, sfreq=sfreq, band_limits=band_limits,
        agg_func=agg_mode)

    meta_features = generate_meta_features(
        file_name=file_name, agg_func=agg_mode, n_crops=len(crops))

    phase_features = generate_phase_features(
        signals=signals, agg_func=agg_mode, band_limits=band_limits,
        outlier_mask=outlier_mask, sfreq=sfreq,
        epoch_duration_s=epoch_duration_s)

    time_features = generate_time_features(
        crops=crops, agg_func=agg_mode, sfreq=sfreq)

    all_features = [features for features in
                    [cwt_features, dwt_features, dft_features, meta_features,
                     phase_features, time_features] if features is not None]
    # concatenate recording features or crop features
    axis = 1 if agg_mode in ["none", "None", None] else 0
    features = np.concatenate(all_features, axis=axis)
    return features
