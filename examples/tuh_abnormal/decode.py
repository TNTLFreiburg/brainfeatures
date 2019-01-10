from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import OrderedDict
from sklearn.svm import SVC, SVR
import pandas as pd
import numpy as np
import logging
import time
import os

from brainfeatures.utils.sun_grid_engine_util import parse_run_args
from brainfeatures.data_set.tuh_abnormal import TuhAbnormal
from brainfeatures.experiment.experiment import Experiment
from brainfeatures.utils.file_util import json_store


def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight, multioutput))


# TODO: add to cropped features as well
# def add_meta_feature(exp):
#     features_to_add = ["age", "gender"]
#     target = exp.data_sets["train"].target
#     features_to_add.remove(target)
#     if "eval" in exp.features:
#         ds = exp.data_sets["eval"]
#         ages, genders = ds.ages, ds.genders
#         if "age" in features_to_add:
#             exp.features["eval"] = [np.append(fv, age) for fv, age in zip(exp.features["eval"], ages)]
#         if "gender" in features_to_add:
#             genders = [0 if gender == "M" else 1 for gender in genders]
#             exp.features["eval"] = [np.append(fv, gender) for fv, gender in zip(exp.features["eval"], genders)]
#         # if "pathological" in features_to_add:
#         #     pathologicals = ds.pathologicals
#         #     exp.features["eval"] = [np.append(fv, pathological) for fv, pathological in zip(exp.features["eval"], pathologicals)]
#     else:
#         ds = exp.data_sets["train"]
#         ages, genders = ds.ages, ds.genders
#         if "age" in features_to_add:
#             exp.features["train"] = [np.append(fv, age) for fv, age in zip(exp.features["train"], ages)]
#             exp.feature_labels.append("meta_age")
#         if "gender" in features_to_add:
#             genders = [0 if gender == "M" else 1 for gender in genders]
#             exp.features["train"] = [np.append(fv, gender) for fv, gender in zip(exp.features["train"], genders)]
#             exp.feature_labels.append("meta_gender")
#         # if "pathological" in features_to_add:
#         #     pathologicals = ds.pathologicals
#         #     exp.features["train"] = [np.append(fv, pathological) for fv, pathological in zip(exp.features["train"], pathologicals)]


# TODO: add to cropped features as well
def add_meta_feature(data_set, features, feature_labels):
    genders = data_set.genders
    assert len(np.unique(genders)) == 2
    assert "F" in genders and "M" in genders
    genders = [0 if gender == "M" else 1 for gender in genders]
    features_to_add = OrderedDict([
        ("age", data_set.ages),
        ("gender", genders),
    ])
    target = data_set.target
    if target in features_to_add:
        features_to_add.pop(target)
    for feature in features_to_add:
        for i in range(len(features)):
            repeated_meta_feature = np.repeat(genders[i], len(features[i]))
            repeated_meta_feature = repeated_meta_feature.reshape(-1, 1)
            features[i] = np.concatenate((features[i], repeated_meta_feature), axis=1)
        feature_label = "meta_" + feature
        if feature_label not in feature_labels[::-1]:
            feature_labels.append(feature_label)
    return features, feature_labels


def run_exp(train_dir, eval_dir, model, n_folds_or_repetitions,
            n_jobs, n_recordings, task, C, gamma, bootstrap,
            min_samples_split, min_samples_leaf, criterion, max_depth, max_features,
            n_estimators, result_dir, feature_vector_modifier=add_meta_feature):

    train_set_feats = TuhAbnormal(train_dir, target=task, n_recordings=n_recordings,
                                  subset="train", extension=".h5")
    train_set_feats.load()

    eval_set_feats = None
    if eval_dir is not None:
        eval_set_feats = TuhAbnormal(eval_dir, target=task,
                                     subset="eval", extension=".h5")
        eval_set_feats.load()

    if model == "rf":
        if task == "age":
            clf = RandomForestRegressor(
                bootstrap=bootstrap,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                n_estimators=n_estimators,
                criterion=criterion,
                n_jobs=n_jobs
            )
        else:
            clf = RandomForestClassifier(
                bootstrap=bootstrap,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                n_estimators=n_estimators,
                criterion=criterion,
                n_jobs=n_jobs
            )
    else:
        assert model == "svm", "unknown model"
        if task == "age":
            clf = SVR(
                C=C,
                gamma=gamma
            )
        else:
            clf = SVC(
                C=C,
                gamma=gamma
            )

    if task == "age":
        metrics = [root_mean_squared_error]
    else:
        metrics = [accuracy_score, roc_auc_score]

    exp = Experiment(
        train_set=train_set_feats,
        clf=clf,
        cleaning_procedure=None,
        n_splits_or_repetitions=n_folds_or_repetitions,
        feature_generation_params=None,
        feature_generation_procedure=None,
        n_jobs=n_jobs,
        metrics=metrics,
        eval_set=eval_set_feats,
        feature_vector_modifier=feature_vector_modifier,
    )
    exp.run()
    return exp


def write_kwargs(kwargs):
    result_dir = kwargs["result_dir"]
    # save predictions and rf feature importances for further analysis
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    json_store(kwargs, result_dir + "config.json")


def make_final_predictions():
    result_dir = kwargs["result_dir"]
    if kwargs["eval_dir"] is None:
        set_name = "train"
    else:
        set_name = "eval"

    exp.predictions[set_name].to_csv(result_dir + "predictions_" + set_name + ".csv")
    if kwargs["model"] == "rf":
        importances_by_fold = pd.DataFrame()
        for i, i_info in enumerate(exp.info[set_name]):
            feature_importances = i_info["feature_importances"]
            importances_df = create_df_from_feature_importances(i, feature_importances)
            importances_by_fold = importances_by_fold.append(importances_df)

        importances_by_fold.to_csv(result_dir + "feature_importances_" + set_name + ".csv")


def create_df_from_feature_importances(id_, importances):
    """ create a pandas data frame from predictions and labels to an id"""
    df = pd.DataFrame()
    repeated_id = np.repeat(id_, len(importances))
    for i in range(len(repeated_id)):
        row = {"id": repeated_id[i],
               "importances": importances[i]}
        df = df.append(row, ignore_index=True)
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    kwargs = parse_run_args()
    start_time = time.time()
    exp = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    logging.info("Experiment runtime: {:.2f} sec".format(run_time))

    write_kwargs(kwargs)
    make_final_predictions()
