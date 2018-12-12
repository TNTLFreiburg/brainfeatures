import numpy as np
import resampy

from brainfeaturedecode.utils.file_util import get_duration_with_raw_mne


def reject_too_long_recording(file_, max_recording_mins):
    duration = get_duration_with_raw_mne(file_)
    return duration > max_recording_mins * 60, duration


def remove_start(signals, sec_to_cut_start, fs):
    return signals[:, int(sec_to_cut_start * fs):]


def remove_end(signals, sec_to_cut_end, fs):
    return signals[:, :-int(sec_to_cut_end * fs)]


def take_part(signals, duration_recording_mins, fs):
    return signals[:, :int(duration_recording_mins * 60 * fs)]


def resample(signals, fs, resample_fs):
    return resampy.resample(x=signals, sr_orig=fs, sr_new=resample_fs, axis=1,
                            filter='kaiser_fast')


def clip_values(signals, max_abs_value):
    return np.clip(signals, -max_abs_value, max_abs_value)
