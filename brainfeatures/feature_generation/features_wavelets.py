import numpy as _np


def bounded_variation(coefficients, axis):
    diffs = _np.diff(coefficients, axis=axis)
    abs_sums = _np.sum(_np.abs(diffs), axis=axis)
    max_c = _np.max(coefficients, axis=axis)
    min_c = _np.min(coefficients, axis=axis)
    return _np.divide(abs_sums, max_c - min_c)


def maximum(coefficients, axis):
    return _np.max(coefficients, axis=axis)


def mean(coefficients, axis):
    return _np.mean(coefficients, axis=axis)


def minimum(coefficients, axis):
    return _np.min(coefficients, axis=axis)


def power(coefficients, axis):
    return _np.sum(coefficients*coefficients, axis=axis)


def power_ratio(powers, axis=-2):
    ratios = powers / _np.sum(powers, axis=axis, keepdims=True)
    return ratios


def spectral_entropy(ratios, axis=None):
    return -1 * ratios * _np.log(ratios)


def variance(coefficients, axis):
    return _np.var(coefficients, axis=axis)
