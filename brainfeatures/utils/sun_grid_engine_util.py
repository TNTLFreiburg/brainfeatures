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


def custom_parse(variables, arg):
    try:
        variables[arg] = int(variables[arg])
    except:
        variables[arg] = None
    return variables


def parse_run_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", required=True, type=int)
    parser.add_argument('--bootstrap', dest='bootstrap', action='store_true')
    parser.add_argument('--no-bootstrap', dest='bootstrap', action='store_false')
    parser.add_argument("--criterion", required=True, type=str)
    parser.add_argument("--eval_dir", required=True, type=str)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--max_depth", required=True, type=int)
    parser.add_argument("--max_features", required=True, type=str)  # add own parsing
    parser.add_argument("--min_samples_leaf", required=True, type=int)
    parser.add_argument("--min_samples_split", required=True, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--n_estimators", required=True, type=int)
    parser.add_argument("--n_folds_or_repetitions", required=True, type=int)
    parser.add_argument("--n_jobs", required=True, type=int)
    parser.add_argument("--n_recordings", required=True, type=str)  # add own parsing
    parser.add_argument("--result_dir", required=True, type=str)
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--train_dir", required=True, type=str)

    known, unknown = parser.parse_known_args()
    if unknown:
        print(unknown)
        print("exiting due to unkown args")
        exit()

    known_vars = vars(known)
    custom_parse_args = ["n_recordings", "max_features"]
    for arg in custom_parse_args:
        known_vars = custom_parse(known_vars, arg)

    if known_vars["eval_dir"] in ["nan", "None"]:
        known_vars["eval_dir"] = None

    return known_vars
