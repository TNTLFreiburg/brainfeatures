from datetime import datetime, date
import pandas as pd
import logging

from brainfeatures.feature_generation.generate_features import \
    generate_features_of_one_file
from brainfeatures.utils.file_util import pandas_store_as_h5
from brainfeatures.data_set.tuh_abnormal import TuhAbnormal
from brainfeatures.utils.sun_grid_engine_util import \
    determime_curr_file_id


def load_one_file_and_info_from_data_set(data_set, file_id):
    file_name = data_set.file_names[file_id]
    signals, sfreq, pathological = data_set[file_id]
    age = data_set.ages[file_id]
    gender = data_set.genders[file_id]
    return signals, sfreq, pathological, age, gender, file_name


def process_one_file(data_set, file_id, out_dir, epoch_duration_s,
                     max_abs_val, window_name, band_limits, channel_names,
                     agg_mode, discrete_wavelet, continuous_wavelet,
                     band_overlap, domains):
    signals, sfreq, pathological, age, gender, file_name = \
        load_one_file_and_info_from_data_set(data_set, file_id)

    feature_df = generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, channel_names, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, domains)

    additional_info = {
        "sfreq": sfreq,
        "pathological": pathological,
        "age": age,
        "gender": gender,
        "n_samples": signals.shape[1],
        "id": file_id,
        "n_windows": len(feature_df),
        "n_features": len(feature_df.columns),
        "agg": agg_mode,
        "name": file_name,
    }
    info_df = pd.DataFrame(additional_info, index=[0])

    pandas_store_as_h5(out_dir+"{:04d}.h5".format(file_id), feature_df, "features")
    pandas_store_as_h5(out_dir+"{:04d}.h5".format(file_id), info_df, "info")


def generate_features_main():
    log = logging.getLogger()
    log.setLevel("INFO")
    today, now = date.today(), datetime.time(datetime.now())
    logging.info('started on {} at {}'.format(today, now))

    train_or_eval = "eval"
    in_dir = "/data/schirrmr/gemeinl/tuh-abnormal-eeg/clean/full/resampy0.2.1_clipafter/v2.0.0/edf/"+train_or_eval+"/"
    out_dir = in_dir.replace("clean", "feats/unagged")
    max_abs_val = 800
    epoch_duration_s = 6
    window_name = "blackmanharris"
    band_limits = [[0, 2], [2, 4],  [4, 8], [8, 13],
                   [13, 18],  [18, 24], [24, 30], [30, 50]]
    channel_names = sorted(['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7',
                            'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4',
                            'PZ', 'T3', 'T4', 'T5', 'T6'])
    agg_mode = None
    discrete_wavelet = "db4"
    continuous_wavelet = "morl"
    band_overlap = True
    domains = "all"
    run_on_cluster = True

    tuh_abnormal = TuhAbnormal(in_dir, ".h5", subset=train_or_eval)
    tuh_abnormal.load()
    # use this to run on cluster. otherwise just give the id of the file that
    # should be cleaned
    if run_on_cluster:
        file_id = determime_curr_file_id(tuh_abnormal, file_id=None)

        if type(file_id) is not int:
            logging.error(file_id)
            exit()

        process_one_file(tuh_abnormal, file_id, out_dir,
                         epoch_duration_s, max_abs_val, window_name,
                         band_limits, channel_names, agg_mode,
                         discrete_wavelet, continuous_wavelet, band_overlap,
                         domains)

    else:
        for file_id in range(len(tuh_abnormal)):
            process_one_file(tuh_abnormal, file_id, out_dir,
                             epoch_duration_s, max_abs_val, window_name,
                             band_limits, channel_names, agg_mode,
                             discrete_wavelet, continuous_wavelet,
                             band_overlap, domains)

    today, now = date.today(), datetime.time(datetime.now())
    logging.info('finished on {} at {}'.format(today, now))


if __name__ == "__main__":
    generate_features_main()
