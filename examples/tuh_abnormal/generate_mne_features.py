from mne_features.univariate import get_univariate_funcs
from mne_features.bivariate import get_bivariate_funcs
from datetime import datetime, date
import logging

from brainfeatures.feature_generation.generate_mne_features import \
    generate_mne_features_of_one_file, default_mne_feature_generation_params
from brainfeatures.utils.file_util import numpy_store, json_store, \
    replace_extension
from brainfeatures.data_set.tuh_abnormal import TuhAbnormal
from brainfeatures.utils.sun_grid_engine_util import \
    determime_curr_file_id


def generate_mne_features_and_info_for_one_file(signals, sfreq,
                                                epoch_duration_s, max_abs_val,
                                                agg_mode):
    selected_funcs = get_univariate_funcs(sfreq)
    selected_funcs.update(get_bivariate_funcs(sfreq))
    feature_vector, feature_labels = generate_mne_features_of_one_file(
        signals, sfreq, selected_funcs,
        default_mne_feature_generation_params, epoch_duration_s,
        max_abs_val, agg_mode)
    return feature_vector, {"feature_labels": feature_labels}


def store_mne_features_and_info(signals, info, in_dir, out_dir, file_name):
    file_name = file_name.replace(in_dir, out_dir)
    file_name = replace_extension(file_name, ".npy")
    numpy_store(file_name, signals)
    new_file_name = replace_extension(file_name, ".json")
    json_store(info, new_file_name)


def load_one_file_and_info_from_data_set(data_set, file_id):
    file_name = data_set.file_names[file_id]
    signals, sfreq, pathological = data_set[file_id]
    age = data_set.ages[file_id]
    gender = data_set.genders[file_id]
    return signals, sfreq, pathological, age, gender, file_name


def process_one_file(data_set, file_id, in_dir, out_dir, epoch_duration_s,
                     max_abs_val, agg_mode):
    signals, sfreq, pathological, age, gender, file_name = \
        load_one_file_and_info_from_data_set(data_set, file_id)
    feature_vector, info = generate_mne_features_and_info_for_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, agg_mode)
    additional_info = {
        "sfreq": sfreq,
        "pathological": pathological,
        "age": age,
        "gender": gender,
        "n_samples": signals.shape[1],
        "n_features": len(feature_vector)
    }
    info.update(additional_info)
    store_mne_features_and_info(feature_vector, info, in_dir, out_dir, file_name)


def generate_mne_features_main():
    log = logging.getLogger()
    log.setLevel("INFO")
    today, now = date.today(), datetime.time(datetime.now())
    logging.info('started on {} at {}'.format(today, now))

    in_dir = "/data/schirrmr/gemeinl/tuh-abnormal-eeg/clean/full/resampy0.2.1_clipafter/v2.0.0/edf/train/"
    out_dir = in_dir.replace("clean", "mne_feats/unagged")
    max_abs_val = 800
    epoch_duration_s = 6
    agg_mode = None
    run_on_cluster = True

    tuh_abnormal = TuhAbnormal(in_dir, ".h5")
    tuh_abnormal.load()
    # use this to run on cluster. otherwise just give the id of the file that
    # should be cleaned
    if run_on_cluster:
        file_id = determime_curr_file_id(tuh_abnormal, file_id=None)

        if type(file_id) is not int:
            logging.error(file_id)
            exit()

        process_one_file(tuh_abnormal, file_id, in_dir, out_dir,
                         epoch_duration_s, max_abs_val, agg_mode)

    else:
        for file_id in range(len(tuh_abnormal)):
            process_one_file(tuh_abnormal, file_id, in_dir, out_dir,
                             epoch_duration_s, max_abs_val, agg_mode)

    today, now = date.today(), datetime.time(datetime.now())
    logging.info('finished on {} at {}'.format(today, now))


if __name__ == "__main__":
    generate_mne_features_main()
