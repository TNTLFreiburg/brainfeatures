from collections import OrderedDict
from glob import glob
import logging
import re

import pandas as pd
import numpy as np

from brainfeatures.data_set.abstract_data_set import DataSet
from brainfeatures.utils.file_util import (natural_key,
                                           parse_age_and_gender_from_edf_header,
                                           mne_load_signals_and_fs_from_edf,
                                           property_in_path)
from brainfeatures.preprocessing.rules import reject_too_long_recording


# check whether this can be replaced by natural key
def _session_key(string):
    """ sort the file name by session """
    p = r'(s\d*)_'
    return re.findall(p, string)


def _time_key(file_name):
    """ provides a time-based sorting key """
    # the splits are specific to tuh abnormal eeg data set
    splits = file_name.split('/')
    p = r'(\d{4}_\d{2}_\d{2})'
    [date] = re.findall(p, splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = _session_key(splits[-2])
    return date_id + session_id + recording_id


def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in
    subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21
        (machine 1, 12, 2, 21) or by time since this is
        important for cv. time is specified in the edf file names
    """
    assert key in ["natural", "time"], "unknown sorting key"
    file_paths = glob(path + '**/*' + extension, recursive=True)
    if key == "time":
        sorting_key = _time_key
    else:
        sorting_key = natural_key
    file_names = sorted(file_paths, key=sorting_key)

    assert len(file_names) > 0, ("something went wrong. Found no {} files in {}"
                                 .format(extension, path))
    return file_names


class TuhAbnormal(DataSet):
    """tuh abnormal data set. file names are given as"""
    # v2.0.0/edf/eval/abnormal/01_tcp_ar/007/00000768/s003_2012_04_06/
    # 00000768_s003_t000.edf
    def __init__(self, data_path, extension, subset="train", channels=sorted([
        'A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2',
        'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']),
                 key="time", n_recordings=None, target="pathological",
                 max_recording_mins=None, ch_name_pattern="EEG {}-REF"):
        self.max_recording_mins = max_recording_mins
        self.ch_name_pattern = ch_name_pattern
        self.n_recordings = n_recordings
        self.extension = extension
        self.data_path = data_path
        self.channels = channels
        self.target = target
        self.subset = subset
        self.key = key
        self.gender_int_map = {"M": 0, "F": 1}

        self.pathologicals = []
        self.file_names = []
        self.genders = []
        self.targets = []
        self.sfreqs = []
        self.ages = []

        assert data_path.endswith("/"), "data path has to end with '/'"
        assert extension.startswith("."), "extension has to start with '.'"
        if self.subset == "eval":
            assert self.max_recording_mins is None, ("do not reject eval "
                                                     "recordings")

    def load(self):
        # read all file names in path with given extension sorted by key
        self.file_names = read_all_file_names(
            self.data_path, self.extension, self.key)

        assert self.subset in self.file_names[0], (
            "cannot parse {} from file name {}"
            .format(self.subset, self.file_names[0]))

        # prune this file names to train or eval subset
        self.file_names = [file_name for file_name in self.file_names
                           if self.subset in file_name.split('/')]

        n_picked_recs = 0
        files_to_delete = []
        for file_name in self.file_names:
            if self.n_recordings is not None:
                if n_picked_recs == self.n_recordings:
                    break

            # if this is raw version of data set, reject too long recordings
            if self.extension == ".edf":
                if self.max_recording_mins is not None:
                    # reject recordings that are too long
                    rejected, duration = reject_too_long_recording(
                        file_name, self.max_recording_mins)
                    if rejected:
                        files_to_delete.append(file_name)
                        continue
            n_picked_recs += 1

            assert self.target in ["pathological", "age", "gender"], (
                "unknown target {}".format(self.target))
            assert self.extension in [".edf", ".h5"], (
                "unknown file format {}".format(self.extension))
            if self.extension == ".edf":
                # get pathological status, age and gender for edf file
                pathological = property_in_path(file_name, "abnormal")
                age, gender = parse_age_and_gender_from_edf_header(file_name)
            else:
                info_df = pd.read_hdf(file_name, key="info")
                assert len(info_df) == 1, "too many rows in info df"
                info = info_df.iloc[-1].to_dict()
                pathological = info["pathological"]
                age = info["age"]
                gender = info["gender"]
                self.sfreqs.append(int(info["sfreq"]))

            # encode gender string as integer
#            assert gender in ["M", "F"], "unknown gender"
            if gender in self.gender_int_map.keys():
                gender = self.gender_int_map[gender]
            else:
                assert gender in self.gender_int_map.values(), "unknown gender"
#            assert gender in self.gender_int_map.keys() or gender in self.gender_int_map.values(), "unknown gender"
#            gender = 0 if gender == "M" else 1
 #           gender = self.gender_int_map[gender]

            targets = {"pathological": pathological, "age": age,
                       "gender": gender}
            self.targets.append(targets[self.target])
            self.ages.append(age)
            self.genders.append(gender)
            self.pathologicals.append(pathological)

        if self.max_recording_mins is not None:
            # prune list of all file names to n_recordings
            for file_name in files_to_delete:
                self.file_names.remove(file_name)

        if self.n_recordings is not None:
            self.file_names = self.file_names[:self.n_recordings]

        assert len(self.file_names) == len(self.targets), "lengths differ"
        if self.n_recordings is not None:
            assert len(self.file_names) == self.n_recordings, (
                "less recordings picked than desired")
        assert len(np.intersect1d(self.file_names, files_to_delete)) == 0, (
            "deleting unwanted file names failed")

    def __getitem__(self, index):
        file_ = self.file_names[index]
        label = self.targets[index]
        # raw tuh data
        if self.extension == ".edf":
            signals, sfreq = mne_load_signals_and_fs_from_edf(
                file_, self.channels, self.ch_name_pattern)
        # preprocessed tuh data / features
        else:
            assert self.extension == ".h5", "unknown data format"
            signals = pd.read_hdf(file_, key="data")
            x_dim, y_dim = signals.shape
            if x_dim > y_dim:
                signals = signals.T
            sfreq = self.sfreqs[index]
        return signals, sfreq, label

    def __len__(self):
        return len(self.file_names)


class TuhAbnormalSubset(DataSet):
    def __init__(self, dataset, indeces):
        self.file_names = [dataset.file_names[i] for i in indeces]
        self.targets = [dataset.targets[i] for i in indeces]
        self.target = dataset.target
        self.sfreqs = [dataset.sfreqs[i] for i in indeces]
        self.ages = [dataset.ages[i] for i in indeces]
        self.genders = [dataset.genders[i] for i in indeces]
        self.pathologicals = [dataset.pathologicals[i] for i in indeces]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_ = self.file_names[idx]
        label = self.targets[idx]
        signals = pd.read_hdf(file_, key="data")
        x_dim, y_dim = signals.shape
        if x_dim > y_dim:
            signals = signals.T
        sfreq = self.sfreqs[idx]
        return signals, sfreq, label


# this function is called once for devel, once for eval set.
# on second call don't add feature name, but add metafeature!
def add_meta_feature(data_set, features, feature_labels):
    """ modify the feature vectors of a data set. here, we add additional
     meta features age and gender """
    features_to_add = OrderedDict([
        ("age", data_set.ages),
        ("gender", data_set.genders),
    ])
    target = data_set.target
    if target in features_to_add:
        features_to_add.pop(target)
    logging.info("now adding {} to feature vectors".
                 format(' and '.join(features_to_add.keys())))
    for feature in features_to_add:
        feature_label = "meta_" + feature
        for i in range(len(features)):
            repeated_meta_feature = np.repeat(features_to_add[feature][i],
                                              len(features[i]))
            repeated_meta_feature = pd.DataFrame(
                repeated_meta_feature.reshape(-1, 1),
                columns=[feature_label])
            features[i] = pd.concat((features[i], repeated_meta_feature),
                                    axis=1)

        if feature_label in feature_labels[::-1]:
            continue
        else:
            feature_labels.append(feature_label)
    return features, feature_labels
