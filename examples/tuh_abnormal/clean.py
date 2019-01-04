from datetime import datetime, date
import logging
import resampy

from brainfeatures.utils.file_util import h5_store, json_store, \
    replace_extension
from brainfeatures.cleaning.clean_raw import clean_one_file
from brainfeatures.data_set.tuh_abnormal import TuhAbnormal
from brainfeatures.utils.sun_grid_engine_util import \
    determime_curr_file_id


def clean_one_file_and_generate_info(signals, sfreq, pathological, age, gender,
                                     sec_to_cut_start, sec_to_cut_end,
                                     duration_recording_mins, resample_freq,
                                     max_abs_val, clip_before_resample):
    signals, resample_freq = clean_one_file(
        signals=signals,
        fs=sfreq,
        sec_to_cut_start=sec_to_cut_start,
        sec_to_cut_end=sec_to_cut_end,
        duration_recording_mins=duration_recording_mins,
        resample_freq=resample_freq,
        max_abs_val=max_abs_val,
        clip_before_resample=clip_before_resample)
    info = {
        "sfreq": resample_freq,
        "pathological": pathological,
        "age": age,
        "gender": gender,
        "n_samples": signals.shape[1]
    }
    return signals, info


def store_clean_signals_and_info(signals, info, in_dir, out_dir, file_name):
    file_name = file_name.replace(in_dir, out_dir)
    new_file_name = replace_extension(path=file_name, new_extension=".json")
    json_store(to_store=info, path=new_file_name)
    new_file_name = replace_extension(path=file_name, new_extension=".h5")
    # save signals as n_samples x n_channels since it reduces time of loading
    # chunks
    h5_store(data=signals.T, path=new_file_name)
    logging.info("wrote clean signals to {}".format(new_file_name))


def load_one_file_and_info_from_data_set(data_set, file_id):
    file_name = data_set.file_names[file_id]
    logging.info("loading {}: {}".format(file_id, file_name))
    signals, sfreq, pathological = data_set[file_id]
    age = data_set.ages[file_id]
    gender = data_set.genders[file_id]
    return signals, sfreq, pathological, age, gender, file_name


def process_one_file(data_set, file_id, in_dir, out_dir, sec_to_cut_start,
                     sec_to_cut_end, duration_recording_mins, resample_freq,
                     max_abs_val, clip_before_resample):
    signals, sfreq, pathological, age, gender, file_name = \
        load_one_file_and_info_from_data_set(
            data_set=data_set,
            file_id=file_id)

    clean_signals, info = clean_one_file_and_generate_info(
        signals=signals,
        sfreq=sfreq,
        pathological=pathological,
        age=age,
        gender=gender,
        sec_to_cut_start=sec_to_cut_start,
        sec_to_cut_end=sec_to_cut_end,
        duration_recording_mins=duration_recording_mins,
        resample_freq=resample_freq,
        max_abs_val=max_abs_val,
        clip_before_resample=clip_before_resample)

    store_clean_signals_and_info(
        signals=clean_signals,
        info=info,
        in_dir=in_dir,
        out_dir=out_dir,
        file_name=file_name)


def clean_main():
    """ runs either one file on cluster (id specified by array job id) or all
    files locally """
    log = logging.getLogger()
    log.setLevel("INFO")
    today, now = date.today(), datetime.time(datetime.now())
    logging.info('started on {} at {}'.format(today, now))

    in_dir = "/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/train/"
    out_dir = in_dir.replace("raw", "clean/full/resampy" + resampy.__version__ + "_clipafter")

    logging.info("reading from {}".format(in_dir))
    logging.info("wrtiting to {}".format(out_dir))

    sec_to_cut_start = 60
    sec_to_cut_end = 0
    duration_recording_mins = 20
    max_recording_mins = 35
    resample_freq = 100
    max_abs_val = 800
    n_recordings = None
    clip_before_resample = False
    run_on_cluster = True

    tuh_abnormal = TuhAbnormal(in_dir, ".edf", n_recordings=n_recordings,
                               max_recording_mins=max_recording_mins)
    tuh_abnormal.load()

    logging.info("there are {} recordings".format(len(tuh_abnormal)))
    # use this to run on cluster. otherwise just give the id of the file that
    # should be cleaned
    if run_on_cluster:
        logging.info("using file id based on sge array job id")
        file_id = determime_curr_file_id(tuh_abnormal, file_id=None)
        assert type(file_id) is int, "type of file_id is {} but has to be an integer".format(type(file_id))
        file_ids = [file_id]
    else:
        file_ids = range(len(tuh_abnormal))
        logging.info("cleaning all files sequentially")

    for file_id in file_ids:
        process_one_file(
            data_set=tuh_abnormal,
            file_id=file_id,
            in_dir=in_dir,
            out_dir=out_dir,
            sec_to_cut_start=sec_to_cut_start,
            sec_to_cut_end=sec_to_cut_end,
            duration_recording_mins=duration_recording_mins,
            resample_freq=resample_freq,
            max_abs_val=max_abs_val,
            clip_before_resample=clip_before_resample)

    today, now = date.today(), datetime.time(datetime.now())
    logging.info('finished on {} at {}'.format(today, now))


if __name__ == "__main__":
    clean_main()
