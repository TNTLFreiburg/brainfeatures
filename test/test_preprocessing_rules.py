import numpy as np

from brainfeatures.preprocessing.rules import reject_too_long_recording, \
    remove_start, remove_end, take_part, resample, clip_values


def test_reject_too_long_recording():
    # TODO: how to test this?
    pass


def test_remove_start():
    fs = 100
    signals = np.random.rand(2*fs*60).reshape(2, -1)
    result = remove_start(signals, 20, fs)
    expected = signals[:, 20*fs:]
    np.testing.assert_array_equal(expected, result)

    result = remove_start(signals, 60, fs)
    expected = np.ndarray((2, 0))
    np.testing.assert_allclose(expected, result)


def test_remove_end():
    fs = 100
    signals = np.random.rand(2*fs*60).reshape(2, -1)
    result = remove_end(signals, 20, fs)
    expected = signals[:, :-20*fs]
    np.testing.assert_array_equal(expected, result)

    result = remove_end(signals, 60, fs)
    expected = np.ndarray((2, 0))
    np.testing.assert_allclose(expected, result)


def test_take_part():
    fs = 100
    signals = np.random.rand(2 * fs * 2 * 60).reshape(2, -1)
    result = take_part(signals, 1, fs)
    np.testing.assert_array_equal(signals[:, :fs * 60], result)

    result = take_part(signals, 2, fs)
    np.testing.assert_array_equal(signals, result)

    result = take_part(signals, 0, fs)
    expected = np.ndarray((2, 0))
    np.testing.assert_array_equal(expected, result)


def test_resample():
    fs = 200
    signals = np.random.rand(2 * fs * 60).reshape(2, -1)
    resample_fs = 100
    result = resample(signals, fs, resample_fs)
    assert signals.shape[-1]/2 == result.shape[-1]


def test_clip_values():
    np.random.seed(0)
    fs = 100
    signals = np.random.rand(2 * fs * 60).reshape(2, -1) - .5
    max_abs_value = .4
    mask = signals > max_abs_value
    assert np.sum(mask) > 0
    mask = signals < -max_abs_value
    assert np.sum(mask) > 0
    result = clip_values(signals, max_abs_value)
    mask = result > max_abs_value
    assert np.sum(mask) == 0
    mask = result < -max_abs_value
    assert np.sum(mask) == 0
