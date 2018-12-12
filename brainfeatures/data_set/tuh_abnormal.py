from glob import glob
import pandas as pd
import numpy as np
import h5py
import re

from brainfeatures.data_set.abstract_data_set import DataSet
from brainfeatures.utils.file_util import natural_key, \
    parse_age_and_gender_from_edf_header, numpy_load, json_load, \
    mne_load_signals_and_fs_from_edf, property_in_path, h5_load, \
    replace_extension
from brainfeatures.cleaning.rules import reject_too_long_recording
# TODO: this sucks. improve! simplify!
# TODO: aggregate features on read?


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


def _read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    assert key in ["natural", "time"], "unknown sorting key"
    file_paths = glob(path + '**/*' + extension, recursive=True)
    if key == "time":
        sorting_key = _time_key
    else:
        sorting_key = natural_key
    file_names = sorted(file_paths, key=sorting_key)

    assert len(file_names) > 0, \
        "something went wrong. Found no {} files in {}".format(
            extension, path)
    return file_names


class TuhFeatures(DataSet):
    def __init__(self, data_path, target="pathological", data_key="features", info_key="info",
                 n_recordings=None):
        super(TuhFeatures, self).__init__()
        self.data_path = data_path
        self.target = target
        self.file_names = []
        self.feature_labels = None
        self.orig_file_names = []
        self.pathologicals = []
        self.genders = []
        self.targets = []
        self.ages = []
        self.data_key = data_key
        self.info_key = info_key
        self.n_recordings=n_recordings

    def load(self):
        self.file_names = _read_all_file_names(self.data_path, "h5", key="natural")
        if self.n_recordings is not None:
            self.file_names = self.file_names[:self.n_recordings]
        self.orig_file_names = [None] * len(self.file_names)
        self.pathologicals = [None] * len(self.file_names)
        self.genders = [None] * len(self.file_names)
        self.ages = [None] * len(self.file_names)
        self.targets = [None] * len(self.file_names)

    def get_raw(self, index):
        file_name = self.file_names[index]
        features_df = pd.read_hdf(file_name, key=self.data_key)
        if self.info_key:
            info_df = pd.read_hdf(file_name, key=self.info_key)
            return features_df, info_df
        return features_df, None

    def __getitem__(self, index):
        file_ = self.file_names[index]
        x = pd.read_hdf(file_, key=self.data_key)
        feature_labels = list(x.columns)
        x = np.array(x).squeeze()
        if self.feature_labels is None:
            self.feature_labels = feature_labels

        if self.info_key:
            info = pd.read_hdf(file_, key=self.info_key)
            info = info.iloc[0]
            fs = info["sfreq"]
            y = info[self.target]

            self.orig_file_names[index] = info["name"]
            self.pathologicals[index] = info["pathological"]
            self.targets[index] = info[self.target]
            self.genders[index] = info["gender"]
            self.ages[index] = info["age"]
            return x, fs, y
        return x, None, None

    def __len__(self):
        return len(self.file_names)


class TuhAbnormal(DataSet):
    """tuh abnormal data set. file names are given as"""
    # v2.0.0/edf/eval/abnormal/01_tcp_ar/007/00000768/s003_2012_04_06/00000768_s003_t000.edf
    def __init__(self, data_path, extension, subset="train", channels=sorted([
        'A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2',
        'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']),
                 key="time", n_recordings=None, target="pathological",
                 max_recording_mins=None, ch_name_pattern="EEG {}-REF"):
        super(TuhAbnormal, self).__init__()
        self.max_recording_mins = max_recording_mins
        self.ch_name_pattern = ch_name_pattern
        self.n_recordings = n_recordings
        self.extension = extension
        self.data_path = data_path
        self.channels = channels
        self.target = target
        self.subset = subset
        self.key = key

        self.pathologicals = []
        self.file_names = []
        self.genders = []
        self.targets = []
        self.sfreqs = []
        self.ages = []

        if self.subset == "eval":
            assert self.max_recording_mins is None, "do not reject eval recordings"

    def load(self):
        # read all file names in path with given extension sorted by key
        self.file_names = _read_all_file_names(
            self.data_path, self.extension, self.key)

        assert self.subset in self.file_names[0], \
            "cannot parse train or eval from file name {}"\
                .format(self.file_names[0])

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

            assert self.target in ["pathological", "age", "gender"], \
                "unknown target {}".format(self.target)
            assert self.extension in [".edf", ".npy", ".h5"], \
                "unknown file format {}".format(self.extension)
            if self.extension == ".edf":
                # get pathological status, age and gender for edf file
                pathological = property_in_path(file_name, "abnormal")
                age, gender = parse_age_and_gender_from_edf_header(file_name)
            else:
                # load info json file of clean recording
                # get pathological status, age, gender and sfreq for clean file
                new_file_name = replace_extension(file_name, ".json")
                info = json_load(new_file_name)
                pathological = info["pathological"]
                age = int(info["age"])
                gender = info["gender"]
                self.sfreqs.append(info["sfreq"])
            # else:
            #     df = pd.read_hdf(file_name)
            #     pathological = df["pathological"]
            #     age = df["age"]
            #     gender = df["gender"]
            #     self.sfreqs.append(df["sfreq"])

            targets = {"pathological": pathological, "age": age, "gender": gender}
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
            assert len(self.file_names) == self.n_recordings, \
                "less recordings picked than desired"
        assert len(np.intersect1d(self.file_names, files_to_delete)) == 0, \
            "deleting unwanted file names failed"

    def __getitem__(self, index):
        file_ = self.file_names[index]
        label = self.targets[index]
        if self.extension == ".edf":
            signals, sfreq = mne_load_signals_and_fs_from_edf(
                file_, self.channels, self.ch_name_pattern)
            return signals, sfreq, label
        elif self.extension == ".npy":
            data = numpy_load(file_)
            return data, self.sfreqs[index], label
        elif self.extension == ".h5":
            data = h5_load(file_)
            return data, self.sfreqs[index], label

    def __len__(self):
        return len(self.file_names)


class TuhAbnormalTrain(TuhAbnormal):
    """ train subset of tuh abnormal"""
    def __init__(self, data_path, extension, target, max_recording_mins,
                 n_recordings=None, key="time"):
        TuhAbnormal.__init__(self, data_path=data_path, extension=extension,
                             key=key, n_recordings=n_recordings, target=target,
                             max_recording_mins=max_recording_mins,
                             subset="train")


class TuhAbnormalEval(TuhAbnormal):
    """ eval subset of tuh abnormal"""
    def __init__(self, data_path, extension, target, max_recording_mins,
                 n_recordings=None, key="time"):
        TuhAbnormal.__init__(self, data_path=data_path, extension=extension,
                             key=key, n_recordings=n_recordings, target=target,
                             max_recording_mins=max_recording_mins,
                             subset="eval")
