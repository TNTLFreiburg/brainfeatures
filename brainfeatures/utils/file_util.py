from mne.io import read_raw_edf
import pandas as pd
import numpy as np
import h5py
import json
import os
import re


def replace_extension(path, new_extension):
    assert new_extension.startswith(".")
    old_exension = os.path.splitext(path)[1]
    path = path.replace(old_exension, new_extension)
    return path


def json_store(to_store, path):
    """ store sth to json file """
    assert path.endswith(".json"), "wrong file extension"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as json_file:
        json.dump(to_store, json_file, indent=4, sort_keys=True)


def json_load(path):
    """ load sth from json file """
    assert os.path.exists(path), "file not found {}".format(path)
    with open(path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded


def numpy_store(path, data):
    """ store sth to .npy file """
    assert path.endswith(".npy"), "wrong file extension"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.save(path, data)


def numpy_load(path):
    """ load a numpy file. make sure it exists """
    assert os.path.exists(path), "file not found {}".format(path)
    x = np.load(path)
    if len(x.shape) == 2:
        (xdim, ydim) = x.shape
        if xdim > ydim:
            x = x.T
    return x.astype(np.float64)


def h5_store(path, data):
    assert path.endswith(".h5"), "wrong file extension"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    f = h5py.File(path, "w")
    f["signals"] = data
    f.close()


def h5_load(path):
    """ load signals from h5 """
    assert os.path.exists(path), "file not found {}".format(path)
    f = h5py.File(path, "r")
    x = f["signals"][:]
    f.close()
    if len(x.shape) == 2:
        xdim, ydim = x.shape
        if xdim > ydim:
            x = x.T
    return x.astype(np.float64)


def pandas_store(path, df):
    """ store a pandas df as csv file """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    df.to_csv(path)


def pandas_store_as_h5(path, df, key_):
    """ store a pandas df to h5 """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    df.to_hdf(path, key_)


def pandas_load_from_csv(path):
    """ load a csv file to pandas data frame """
    assert os.path.exists(path), "file not found {}".format(path)
    return pd.DataFrame.from_csv(path)


def mne_load_signals_and_fs_from_edf(file_, wanted_chs, ch_name_pattern=None,
                                     factor=1e6):
    """ read an edf file, pick channels, scale with factor and return signals
    as well as sampling frequency """
    assert os.path.exists(file_), "file not found {}".format(file_)
    raw = read_raw_edf(file_, verbose="error")
    fs = raw.info["sfreq"]
    raw = raw.load_data()
    if ch_name_pattern is not None:
        chs = [ch_name_pattern.format(wanted_elec) for wanted_elec in wanted_chs]
    else:
        chs = wanted_chs
    raw = raw.reorder_channels(chs)
    # achieves two things: asserts that channels are sorted and picked
    # channels are in same order
    assert raw.ch_names == sorted(chs), \
        "actual channel names: {}, wanted channels names: {}".format(
            ', '.join(raw.ch_names), ', '.join(chs))

    signals = raw.get_data()
    if factor:
        signals = signals * factor
    return signals, fs


def get_duration_with_raw_mne(file_path):
    assert os.path.exists(file_path), "file not found {}".format(file_path)
    raw = read_raw_edf(file_path, verbose="error")
    n_sampels = raw._raw_lengths[0]
    sfreq = raw.info["sfreq"]
    return int(n_sampels / sfreq)


def parse_recording_length_from_edf_header(file_path):
    """ some recordings were that huge that simply opening them with mne
    caused the program to crash. therefore, open the edf as bytes and only read
    the header. parse the duration from there and check if the file can safely
    be opened
    """
    assert file_path.endswith(".edf"), "unknown file type"
    assert os.path.exists(file_path), "file not found {}".format(file_path)
    f = open(file_path, 'rb')
    header = f.read(256)
    f.close()
    return int(header[236:244].decode('ascii'))


def parse_age_and_gender_from_edf_header(file_path):
    """ parse sex and age of patient from the patient_id in the header of the
    edf file
    :param file_path: path of the recording
    :return: gender (M, X, F) and age of patient
    """
    assert os.path.exists(file_path), "file not found {}".format(file_path)
    f = open(file_path, 'rb')
    content = f.read(88)
    f.close()
    patient_id = content[8:88].decode('ascii')
    [age] = re.findall("Age:(\d+)", patient_id)
    [gender] = re.findall("\s(\w)\s", patient_id)
    return int(age), gender


def parse_property_from_file_name(curr_file, property):
    """ parse shape, age or gender or pathology status from file name """
    assert os.path.exists(curr_file), "file not found {}".format(curr_file)
    # take everything of file name except data type ending
    tmp_file = curr_file.split('.')[-2]
    tokens = tmp_file.split('_')
    value = tokens[tokens.index(property) + 1]
    return value


def parse_pathological_from_file_name(curr_file):
    """ parse pathology status from file name """
    assert os.path.exists(curr_file), "file not found {}".format(curr_file)
    pathologicals = parse_property_from_file_name(curr_file, "pathological")
    return [True if y == "True" else False for y in pathologicals]


def parse_age_from_file_name(curr_file):
    """ parse age from file name """
    assert os.path.exists(curr_file), "file not found {}".format(curr_file)
    return int(parse_property_from_file_name(curr_file, "age"))


def parse_gender_from_file_name(curr_file):
    assert os.path.exists(curr_file), "file not found {}".format(curr_file)
    return parse_property_from_file_name(curr_file, "gender")


def property_in_path(curr_path, property):
    tokens = curr_path.split("/")
    return property in tokens


def natural_key(string):
    """ provides a human-like sorting key of a string """
    p = r'(\d+)'
    key = [int(t) if t.isdigit() else None for t in re.split(p, string)]
    return key


# is this the same as natural key?
# def session_key(string):
#     """ sort the file name by session """
#     p = r'(s\d*)_'
#     return re.findall(p, string)


def read_feature_results(directory, models, decoding_tasks, decoding_types):
    from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, mean_squared_error
    result_df = pd.DataFrame()
    for model in models:
        for decoding_type in decoding_types:
            for task in decoding_tasks:
                for subset in ["cv", "eval"]:
                    path = os.path.join(directory, model, decoding_type, task, subset)
                    if not os.path.exists(path):
                        print("path does not exist: {}".format(path))
                        continue

                    if subset == "eval":
                        train_or_eval = "eval"
                    else:
                        train_or_eval = "train"
                    df = pd.DataFrame.from_csv(os.path.join(path, "predictions_{}.csv".format(train_or_eval)))

                    # compute some metrics
                    roc_curves, aucs, accs, rmses = [], [], [], []
                    for group, d in df.groupby("id"):
                        if task in ["pathological", "gender"]:
                            auc = roc_auc_score(d.y_true, d.y_pred)
                            aucs.append(auc)

                            roc = roc_curve(d.y_true, d.y_pred)
                            roc_curves.append(roc)

                            acc = accuracy_score(d.y_true, d.y_pred >= .5)
                            accs.append(acc)
                        else:
                            rmse = np.sqrt(mean_squared_error(d.y_true, d.y_pred))
                            rmses.append(rmse)

                    n = len(df.groupby("id"))
                    if task in ["pathological", "gender"]:
                        accs = np.mean(accs) * 100
                        aucs = np.mean(aucs) * 100
                        rmses = None
                    else:
                        accs = None
                        aucs = None
                        rmses = np.mean(rmses)
                    row = {
                        "model": model,
                        "task": task,
                        "accuracy": accs,
                        "auc": aucs,
                        "subset": subset,
                        "rmse": rmses,
                        "n": n,
                        "type": decoding_type,
                    }
                    result_df = result_df.append(row, ignore_index=True)

    return result_df