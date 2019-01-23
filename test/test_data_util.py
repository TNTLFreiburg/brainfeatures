import numpy as np

from brainfeatures.utils.data_util import apply_window_function, \
    split_into_epochs, reject_windows_with_outliers, \
    filter_to_frequency_band, assemble_overlapping_band_limits


def test_split_into_epochs():
    fs = 100
    signals = np.random.rand(2 * 60 * fs).reshape(2, -1)
    result = split_into_epochs(signals, fs, 30)
    expected = np.stack([signals[:, :30*fs], signals[:, 30*fs:]])
    np.testing.assert_array_equal(expected, result)

    result = split_into_epochs(signals, fs, 40)
    expected = signals[:, :fs*40].reshape(1, 2, -1)
    np.testing.assert_array_equal(expected, result)


def test_reject_windows_with_outliers():
    fs = 100
    signals = np.random.rand(2 * 60 * fs).reshape(2, -1)
    epochs = split_into_epochs(signals, fs, 20)
    mask = reject_windows_with_outliers(epochs, 1)
    assert len(epochs) == len(mask)
    assert np.sum(mask) == 0

    epochs[0][0][0] += 1
    epochs[-1][-1][-1] *= -1
    epochs[-1][-1][-1] -= 1
    mask = reject_windows_with_outliers(epochs, 1)
    assert np.sum(mask) == 2
    np.testing.assert_array_equal(epochs[~mask], [epochs[1]])


def test_assemble_overlapping_band_limits():
    band_limits = [[0, 2], [2, 4], [4, 8], [8, 13], [13, 30]]
    result = assemble_overlapping_band_limits(band_limits)
    expected = [[0, 2], [1, 3], [2, 4], [3, 6], [4, 8], [6, 10], [8, 13], [10, 21], [13, 30]]
    np.testing.assert_array_equal(expected, result)


def test_apply_window_function():
    fs = 100
    signals = np.random.rand(2 * 60 * fs).reshape(2, -1)
    epochs = split_into_epochs(signals, fs, 20)
    result = apply_window_function(epochs, "boxcar")
    np.testing.assert_array_equal(epochs, result)

    result = apply_window_function(epochs, "blackmanharris")
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, epochs, result)


# def test_filer_to_frequency_band():
#     TODO: how to test this?
