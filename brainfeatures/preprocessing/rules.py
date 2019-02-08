import numpy as np
import resampy

from brainfeatures.utils.file_util import get_duration_with_raw_mne


def reject_too_long_recording(
        file_: str, max_recording_mins: int) -> (np.ndarray, int):
    duration = get_duration_with_raw_mne(file_)
    return duration > max_recording_mins * 60, duration


def remove_start(
        signals: np.ndarray, sec_to_cut_start: int, fs: int) -> np.ndarray:
    return signals[:, int(sec_to_cut_start * fs):]


def remove_end(
        signals: np.ndarray, sec_to_cut_end: int, fs: int) -> np.ndarray:
    return signals[:, :-int(sec_to_cut_end * fs)]


def take_part(signals: np.ndarray, duration_recording_mins: int, fs: int) -> \
        np.ndarray:
    return signals[:, :int(duration_recording_mins * 60 * fs)]


def resample(signals: np.ndarray, fs: int, resample_fs: int) -> np.ndarray:
    return resampy.resample(x=signals, sr_orig=fs, sr_new=resample_fs, axis=1,
                            filter='kaiser_fast')


def clip_values(signals: np.ndarray, max_abs_value: float) -> np.ndarray:
    return np.clip(signals, -max_abs_value, max_abs_value)
