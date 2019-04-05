from itertools import product
import pandas as pd
import numpy as np
import logging
import pickle
import json
import os
import re

from mne.io import read_raw_edf


def replace_extension(path, new_extension):
    """ replace an extension """
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


def pandas_store_as_h5(path, df, key_):
    """ store a pandas df to h5 """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    df.to_hdf(path, key_)


def mne_load_signals_and_fs_from_edf(file_, wanted_chs, ch_name_pattern=None,
                                     factor=1e6):
    """ read an edf file, pick channels, scale with factor and return signals
    as well as sampling frequency """
    assert os.path.exists(file_), "file not found {}".format(file_)
    raw = read_raw_edf(file_, verbose="error")
    fs = raw.info["sfreq"]
    raw = raw.load_data()
    if ch_name_pattern is not None:
        chs = [ch_name_pattern.format(ch) for ch in wanted_chs]
    else:
        chs = wanted_chs
    raw = raw.reorder_channels(chs)
    # achieves two things: asserts that channels are sorted and picked
    # channels are in same order
    assert raw.ch_names == sorted(chs), (
        "actual channel names: {}, wanted channels names: {}"
            .format(', '.join(raw.ch_names), ', '.join(chs)))

    signals = raw.get_data()
    if factor is not None:
        signals = signals * factor
    signals = pd.DataFrame(signals, index=wanted_chs)
    return signals, fs


def get_duration_with_raw_mne(file_path):
    """ get duration from raw edf mne object without loading it """
    assert os.path.exists(file_path), "file not found {}".format(file_path)
    raw = read_raw_edf(file_path, verbose="error")
    n_sampels = raw._raw_lengths[0]
    sfreq = raw.info["sfreq"]
    return int(n_sampels / sfreq)


def parse_age_and_gender_from_edf_header(file_path, return_raw_header=False):
    """ parse sex and age of patient from the patient_id in the header of the
    edf file
    :param file_path: path of the recording
    :param return_raw_header: whether to return raw header or parse age/gender
    :return: gender (M, X, F) and age of patient
    """
    assert os.path.exists(file_path), "file not found {}".format(file_path)
    f = open(file_path, 'rb')
    content = f.read(88)
    f.close()
    if return_raw_header:
        return content
    patient_id = content[8:88].decode('ascii')
    [age] = re.findall("Age:(\d+)", patient_id)
    [gender] = re.findall("\s(\w)\s", patient_id)
    return int(age), gender


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


def save_exp(exp, save_raw=False, out_dir=None):
    """ save all relevant information contained in an experiment """
    if save_raw:
        with open(out_dir + "exp.pkl", "wb") as pickle_file:
            pickle.dump(exp, pickle_file)
    config = {}
    for i in range(exp._n_runs):
        for subset in exp.predictions.keys():
            preds = exp.predictions[subset]
            if subset in exp.performances.keys() and out_dir is not None:
                performances = exp.performances[subset]
                preds.to_csv(out_dir + "predictions_{}.csv".format(subset, i))
                performances.to_csv(
                    out_dir + "performances_{}.csv".format(subset, i))

            if subset in exp.info.keys() and "feature_importances" in \
                    exp.info[subset].keys():
                feature_importances = exp.info[subset]["feature_importances"]
                if out_dir is not None:
                    feature_importances.to_csv(
                        out_dir + "feature_importances_{}.csv".format(subset, i))

    config.update({"shuffle": exp._shuffle_splits})
    config.update({"n_runs": exp._n_runs})
    config.update({"n_jobs": exp._n_jobs})
    if exp._preproc_params is not None:
        config.update(exp._preproc_params)
    if exp._feat_gen_params is not None:
        config.update(exp._feat_gen_params)
    if exp._pca_thresh is not None:
        config.update({"pca_thresh": exp._pca_thresh})

    for key, value in exp.times.items():
        if type(value) is dict:
            for key2, value2 in value.items():
                config.update({'_'.join(["time", key, key2]): value2})
        else:
            config.update({'_'.join(["time", key]): value})

    config.update({"n_features": len(exp._feature_names)})

    d = {}
    if "estimator_params" in exp._estimator.__dict__:
        params = exp._estimator.__dict__["estimator_params"]
    else:
        params = []
    for param in params:
        d.update(
            {'_'.join(["model", param]): exp._estimator.__dict__[param]})
    config.update(d)
    if "n_estimators" in exp._estimator.__dict__:
        config.update(
            {"n_estimators": exp._estimator.__dict__["n_estimators"]})

    config.update({"sfreq": exp.info["devel"]["sfreq"]})
    config.update(
        {"model": str(exp._estimator.__class__).split('.')[-1][:-2]})

    for param in ["C", "gamma", "kernel"]:
        if param in exp._estimator.__dict__:
            if param == "gamma":
                param = '_' + param
            config.update({'_'.join(["model", param]):
                               exp._estimator.__dict__[param]})

    if out_dir is not None:
        with open(out_dir + "config.json", "w") as json_file:
            json.dump(config, json_file, indent=4, sort_keys=True)
    return config


def read_feature_results(directory, models, decoding_tasks, decoding_types):
    """ read features results from directory structure"""
    from sklearn.metrics import (roc_auc_score, accuracy_score, roc_curve,
                                 mean_squared_error)
    subsets = ["cv", "eval"]
    result_df = pd.DataFrame()
    for model, decoding_type, task, subset in product(models, decoding_types,
                                                      decoding_tasks, subsets):
        path = os.path.join(directory, model, decoding_type, task, subset)
        if not os.path.exists(path):
            logging.error("path does not exist: {}".format(path))
            continue

        if subset == "eval":
            train_or_eval = "eval"
        else:
            train_or_eval = "train"
        df = pd.DataFrame.from_csv(os.path.join(
            path, "predictions_{}.csv".format(train_or_eval)))

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
