from joblib import Parallel, delayed
from datetime import datetime, date
import pandas as pd
import numpy as np
import logging

from brainfeatures.utils.file_util import pandas_store_as_h5, \
    replace_extension
from brainfeatures.preprocessing.preprocess_raw import preprocess_one_file, \
    default_preproc_params
from brainfeatures.data_set.tuh_abnormal import TuhAbnormal
from brainfeatures.utils.sun_grid_engine_util import \
    determime_curr_file_id


def process_one_file(data_set, file_id, in_dir, out_dir, dtype,
                     sec_to_cut_start, sec_to_cut_end, duration_recording_mins,
                     resample_freq, max_abs_val, clip_before_resample):
    file_name = data_set.file_names[file_id]
    logging.info("loading {}: {}".format(file_id, file_name))
    signals, sfreq, pathological = data_set[file_id]
    age = data_set.ages[file_id]
    gender = data_set.genders[file_id]

    channels = signals.index
    signals = np.array(signals)

    preprocessed_signals, resample_freq, _ = preprocess_one_file(
        signals=signals,
        fs=sfreq,
        target=None,
        sec_to_cut_start=sec_to_cut_start,
        sec_to_cut_end=sec_to_cut_end,
        duration_recording_mins=duration_recording_mins,
        resample_freq=resample_freq,
        max_abs_val=max_abs_val,
        clip_before_resample=clip_before_resample)
    preprocessed_df = pd.DataFrame(preprocessed_signals, index=channels,
                                   dtype=dtype)

    # also include sec_to_cut_start, duration_recording_mins etc in
    #  additional info?
    additional_info = {
        "sfreq": resample_freq,
        "pathological": pathological,
        "age": age,
        "gender": gender,
        "n_samples": preprocessed_signals.shape[1],
        "n_samples_raw": signals.shape[1],
    }
    info_df = pd.DataFrame(additional_info, index=[0])

    new_file_name = file_name.replace(in_dir, out_dir)
    new_file_name = replace_extension(new_file_name, ".h5")
    # store as n_times x n_channels since this is faster in lazy loading
    # with cnns
    pandas_store_as_h5(new_file_name, preprocessed_df.T, "data")
    pandas_store_as_h5(new_file_name, info_df, "info")
    logging.info("wrote clean signals to {}".format(new_file_name))


def preprocess_main(data_set, in_dir, out_dir, run_on_cluster, dtype, n_jobs):
    """ runs either one file on cluster (id specified by array job id) or all
    files locally """
    log = logging.getLogger()
    log.setLevel("INFO")
    today, now = date.today(), datetime.time(datetime.now())
    logging.info('started on {} at {}'.format(today, now))

    logging.info("there are {} recordings".format(len(data_set)))
    # use this to run on cluster. otherwise just give the id of the file that
    # should be cleaned
    if run_on_cluster:
        logging.info("using file id based on sge array job id")
        file_id = determime_curr_file_id(data_set, file_id=None)
        assert type(file_id) is int, "type of file_id is {} but has to be an " \
                                     "integer".format(type(file_id))
        file_ids = [file_id]
    else:
        file_ids = range(len(data_set))

    Parallel(n_jobs=n_jobs)(delayed(process_one_file)
                            (data_set, file_id, in_dir, out_dir, dtype,
                             **default_preproc_params) for file_id in file_ids)

    today, now = date.today(), datetime.time(datetime.now())
    logging.info('finished on {} at {}'.format(today, now))


if __name__ == "__main__":
    data_dir = "/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/"
    tuh_abnormal = TuhAbnormal(data_dir, ".edf", n_recordings=None,
                               max_recording_mins=35,
                               subset="train")
    tuh_abnormal.load()
    preprocess_main(
        data_set=tuh_abnormal,
        in_dir=data_dir,
        out_dir=data_dir.replace("raw", "pre"),
        run_on_cluster=False,
        dtype=np.float32,
        n_jobs=1
    )
