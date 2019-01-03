import argparse
import os


def determime_curr_file_id(data_set, file_id=None):
    # if file_id is unset, this means that computation is performed on cluster
    # and hence id is determined by array job id
    if file_id is None:
        try:
            # indexing of sge starts at 1
            file_id = int(os.environ["SGE_TASK_ID"]) - 1
        except KeyError:
            return "cannot find 'SGE_TASK_ID'"
    # if file id is larger than our data set, exit
    if file_id > len(data_set):
        return "cannot have file id {} with {} number of files"\
            .format(file_id, len(data_set))
    # if everything is fine return the integer file id
    return file_id


def parse_run_args():
    # args = [
    #     ['bootstrap', [bool]],
    #     ['C', [float]],
    #     ['criterion', [str]],
    #     ['eval_dir', [str]],
    #     ['gamma', [float]],
    #     ['max_depth', [int]],
    #     ['max_features', [float, str]],
    #     ['meta_feature_id', [int]],
    #     ['min_samples_leaf', [float]],
    #     ['min_samples_split', [float]],
    #     ['model', [str]],
    #     ['n_estimators', [int]],
    #     ['n_folds_or_repetitions', [int]],
    #     ['n_jobs', [int]],
    #     ['n_recordings', [int]],
    #     ['result_dir', [str]],
    #     ['task', [str]],
    #     ['train_dir', [str]]
    # ]
    args = [
        ['bootstrap', bool],
        ['C', float],
        ['criterion', str],
        ['eval_dir', str],
        ['gamma', float],
        ['max_depth', int],
        ['max_features', str],
        ['min_samples_leaf', int],
        ['min_samples_split', int],
        ['model', str],
        ['n_estimators', int],
        ['n_folds_or_repetitions', int],
        ['n_jobs', int],
        ['n_recordings', str],
        ['result_dir', str],
        ['task', str],
        ['train_dir', str]
    ]

    parser = argparse.ArgumentParser()
    for arg, type_ in args:
        parser.add_argument("--" + arg, type=type_, required=False)

    known, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        print("exiting due to unkown args")
        exit()

    known_vars = vars(known)

    try:
        known_vars["n_recordings"] = int(known_vars["n_recordings"])
    except:
        known_vars["n_recordings"] = None

    if known_vars["eval_dir"] in ["nan", "None"]:
        known_vars["eval_dir"] = None

    return known_vars
