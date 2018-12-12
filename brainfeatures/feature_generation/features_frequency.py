import numpy as _np


def maximum(amplitude_spectrum, axis=-1):
    return _np.max(amplitude_spectrum, axis=axis)


def mean(amplitude_spectrum, axis=-1):
    return _np.mean(amplitude_spectrum, axis=axis)


def minimum(amplitude_spectrum, axis=-1):
    return _np.min(amplitude_spectrum, axis=axis)


def peak_frequency(amplitude_spectrum, axis=-1):
    return amplitude_spectrum.argmax(axis=axis)


def power(amplitude_spectrum, axis=-1):
    return _np.sum(amplitude_spectrum * amplitude_spectrum, axis=axis)


def power_ratio(powers, axis=-1):
    ratios = powers / _np.sum(powers, axis=axis, keepdims=True)
    return ratios


def spectral_entropy(ratios, axis=None):
    return -1 * ratios * _np.log(ratios)


def value_range(amplitude_spectrum, axis=-1):
    return _np.ptp(amplitude_spectrum, axis=axis)


def variance(amplitude_spectrum, axis=-1):
    return _np.var(amplitude_spectrum, axis=axis)
