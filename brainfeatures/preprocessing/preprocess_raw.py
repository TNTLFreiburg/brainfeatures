from brainfeatures.preprocessing.rules import (remove_start, remove_end,
                                               take_part, resample, clip_values)


default_preproc_params = {
    "sec_to_cut_start": 60,
    "sec_to_cut_end": 0,
    "duration_recording_mins": 20,
    "resample_freq": 100,
    "max_abs_val": 800,
    "clip_before_resample": False
}


def preprocess_one_file(signals, fs, target, sec_to_cut_start, sec_to_cut_end,
                        duration_recording_mins, resample_freq, max_abs_val,
                        clip_before_resample=False):
    # discard first seconds
    if sec_to_cut_start:
        signals = remove_start(signals, sec_to_cut_start, fs)
    # discard last seconds
    if sec_to_cut_end:
        signals = remove_end(signals, sec_to_cut_end, fs)
    # take at most specied number of remaining minutes
    if duration_recording_mins:
        signals = take_part(signals, duration_recording_mins, fs)

    if clip_before_resample:
        # clip values if desired
        if max_abs_val:
            signals = clip_values(signals, max_abs_val)
        # resample if necessary
        if fs != resample_freq:
            signals = resample(signals, fs, resample_freq)
    else:
        # resample if necessary
        if fs != resample_freq:
            signals = resample(signals, fs, resample_freq)
        # clip values if desired
        if max_abs_val:
            signals = clip_values(signals, max_abs_val)
    return signals, resample_freq, target
