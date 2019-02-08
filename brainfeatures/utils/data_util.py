from mne.filter import filter_data
from scipy import signal
import numpy as np


def split_into_epochs(signals: np.ndarray, sfreq: int,
                      epoch_duration_s: int) -> np.ndarray:
    """ split the signals into non-overlapping epochs """
    n_samples = signals.shape[-1]
    n_samples_in_epoch = int(epoch_duration_s * sfreq)
    epochs = []
    # +1 for last window when n_samples is perfectly dividable
    for i in range(0, n_samples-n_samples_in_epoch+1, n_samples_in_epoch):
        epoch = np.take(signals, range(i, i + n_samples_in_epoch), axis=-1)
        epochs.append(epoch)
    return np.stack(epochs)


def reject_windows_with_outliers(epochs: np.ndarray,
                                 outlier_value: int=800) -> np.ndarray:
    """ reject windows that contain outliers / clipped values """
    pos_outliers = np.sum(epochs >= outlier_value, axis=(1, 2))
    neg_outliers = np.sum(epochs <= -1 * outlier_value, axis=(1, 2))
    outliers = np.logical_or(pos_outliers, neg_outliers)
    return outliers


def apply_window_function(epochs: np.ndarray,
                          window_name: str="blackmanharris") -> np.ndarray:
    """ apply blackmanharris window function """
    assert window_name in ["boxcar", "hamming", "hann", "blackmanharris",
                           "flattop"], \
        "cannot handle window {}".format(window_name)
    n_samples_in_epoch = epochs.shape[-1]
    method_to_call = getattr(signal, window_name)
    window_function = method_to_call(n_samples_in_epoch)
    return epochs * window_function


def filter_to_frequency_band(signals: np.ndarray, sfreq: int, lower: int,
                             upper: int) -> np.ndarray:
    """ filter signals to frequency range defined by upper and lower """
    return filter_data(data=signals, sfreq=sfreq, l_freq=lower, h_freq=upper,
                       verbose='error')


def filter_to_frequency_bands(signals: np.ndarray, bands: list,
                              sfreq: int) -> np.ndarray:
    """ filter signals to frequency ranges defined in bands """
    signals = signals.astype(np.float64)
    (n_signals, n_times) = signals.shape
    band_signals = np.ndarray(shape=(len(bands), n_signals, n_times))
    for band_id, band in enumerate(bands):
        lower, upper = band
        # if lowpass frequency is nyquist frequency, don't make a lowpass
        if upper >= sfreq/2:
            upper = None
        curr_band_signals = filter_to_frequency_band(signals, sfreq, lower,
                                                     upper)
        band_signals[band_id] = curr_band_signals
    return band_signals


def assemble_overlapping_band_limits(non_overlapping_bands: list) -> np.ndarray:
    """ create 50% overlapping frequency bands from non-overlapping bands """
    overlapping_bands = []
    for i in range(len(non_overlapping_bands) - 1):
        band_i = non_overlapping_bands[i]
        overlapping_bands.append(band_i)
        band_j = non_overlapping_bands[i + 1]
        overlapping_bands.append([int((band_i[0] + band_j[0]) / 2),
                                  int((band_i[1] + band_j[1]) / 2)])
    overlapping_bands.append(non_overlapping_bands[-1])
    return np.array(overlapping_bands)
