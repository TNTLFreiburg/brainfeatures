import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition.pca import PCA
# import rfpimp


def get_X_y(data_set, agg_f=None):
    """ read all data from the data set. aggregate if wanted """
    if agg_f is not None:
        assert agg_f is callable, "agg_f has to be a callable"
    X, y = [], []
    for x, sfreq, label in data_set:
        if agg_f is not None and hasattr(x[0][0], "__len__"):
            x = agg_f(x, axis=0)
        elif agg_f is not None and not hasattr(x[0][0], "__len__"):
            logging.warning("Data seems to be aggregated already. "
                            "Skipping aggregation.")
        X.append(x)
        y.append(label)
    y = np.array(y)
    return X, y


def group_X_y(X, y, unique_groups, feature_labels):
    """ split X and y based on groups. create a label and group id for every
    crop"""
    X_grouped, y_grouped, groups = [], [], []
    for trial_i in unique_groups:
        X_grouped.extend(np.array(X[trial_i]))
        y_grouped.extend([y[trial_i]] * len(X[trial_i]))
        groups.extend([trial_i] * len(X[trial_i]))
    X_grouped = pd.DataFrame(X_grouped, columns=feature_labels)
    return X_grouped, y_grouped, groups


def get_train_test(X, y, train_ind, test_ind, epoch_to_group_map):
    """ split data and target wrt given train and test ind, s.t. no crops
    belonging to the same trial are accidentally split
    """
    assert len(X) == len(y), "number of examples and labels does not match"
    assert not (set(train_ind) & set(test_ind)), "train and test set overlap!"
    if hasattr(X[0], "columns"):
        feature_labels = X[0].columns
    else:
        feature_labels = [str(i) for i in range(0, X[0].shape[-1])]

    # do not use np.unique since it also sorts the groups
    # do not use set since it does not have an order
    unique_groups = []
    for group in epoch_to_group_map:
        if group not in unique_groups:
            unique_groups.append(group)
    unique_groups = np.array(unique_groups)

    unique_test_groups = unique_groups[test_ind]
    unique_train_groups = unique_groups[train_ind]
    assert len(unique_groups) == len(X), (
        "length of groups {} and length of data {} does not match".
        format(len(unique_groups), len(X)))

    X_test, y_test, test_groups = group_X_y(X, y, unique_test_groups,
                                            feature_labels)
    X_train, y_train, train_groups = group_X_y(X, y, unique_train_groups,
                                               feature_labels)
    return X_train, y_train, X_test, y_test, train_groups, test_groups


def apply_scaler(X_train, X_test, scaler=StandardScaler()):
    """ fit and transform a train set, transform a test set accordingly """
    feature_labels = X_train.columns
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=feature_labels)
    X_test = pd.DataFrame(X_test, columns=feature_labels)
    return X_train, X_test


def apply_pca(X_train, X_test, pca_thresh):
    """ apply principal component analysis to reduce dimensionality of feature
    vectors"""
    feature_labels = X_train.columns
    pca = PCA(n_components=pca_thresh)
    shape_orig = X_train.shape
    X_train = pca.fit_transform(X_train)
    shape_reduced = X_train.shape
    X_test = pca.transform(X_test)
    logging.info("reduced dimensionality from {} to {}"
                 .format(shape_orig, shape_reduced))
    rows = ["PC-{}".format(i) for i in range(len(pca.components_))]
    pcs = pd.DataFrame(pca.components_, columns=feature_labels, index=rows)
    return X_train, X_test, pcs


def decode_once(X_train, X_test, y_train, y_test, estimator,
                scaler=StandardScaler(), pca_thresh=None, do_importances=True):
    """ take train and test set, maybe apply a scaler or pca, fit train set,
    predict test set, return predictions"""
    logging.debug("{} examples in train set, {} examples in test set".format(
        len(X_train), len(X_test)))
    feature_labels = X_train.columns

    dict_of_dfs = {}
    if scaler is not None:
        X_train, X_test = apply_scaler(X_train, X_test, scaler)
    if pca_thresh is not None:
        X_train, X_test, pcs = apply_pca(X_train, X_test, pca_thresh)
        dict_of_dfs.update({"pca_components": pcs})
        feature_labels = list(pcs.index)
    estimator = estimator.fit(X_train, y_train)
    # TODO: for svm set probability to True?
    if do_importances:
        if hasattr(estimator, "feature_importances_"):
            # save random forest feature importances for analysis
            feature_importances = pd.DataFrame(
                [estimator.feature_importances_], columns=feature_labels)
            dict_of_dfs.update({"feature_importances": feature_importances})

        # rfpimp performances can be applied to any scikit-learn model!
        # rfpimp wants everything as data frame. make sure it gets it
        # rfpimp_importances = rfpimp.importances(
        #     estimator, pd.DataFrame(X_test),
        #     pd.DataFrame(y_test), sort=False)
        # dict_of_dfs.update({"rfpimp_importances": rfpimp_importances.T})
    if hasattr(estimator, "predict_proba"):
        # save probabilities of positive class (equal to 1 - negative class)
        y_hat_train = estimator.predict_proba(X_train)
        y_hat_train = y_hat_train[:, -1]
        y_hat = estimator.predict_proba(X_test)
        y_hat = y_hat[:, -1]
    elif hasattr(estimator, "decision_function"):
        # for auc save svm distance of points to margin
        y_hat = estimator.decision_function(X_test)
        y_hat_train = estimator.decision_function(X_train)
    else:
        # create labels
        y_hat = estimator.predict(X_test)
        y_hat_train = estimator.predict(X_train)

    if hasattr(estimator, "random_state"):
        dict_of_dfs.update({"random_states": [estimator.random_state]})
    return y_hat_train, y_hat, dict_of_dfs


def preds_to_df(id_, predictions, y_true, groups=None):
    """ create a pandas data frame from predictions and labels to an id"""
    predictions_df = pd.DataFrame()
    if groups is not None:
        assert len(groups) == len(y_true), (
            "length mismatch groups: {}, y_true: {}".format(len(groups),
                                                            len(y_true)))
    for i in range(len(y_true)):
        row = {"id": id_,
               "y_true": y_true[i],
               "y_pred": predictions[i]}
        if groups is not None:
            row.update({"group": groups[i]})
        predictions_df = predictions_df.append(row, ignore_index=True)
    return predictions_df


def validate(X_train, y_train, estimator, n_splits, shuffle_splits,
             scaler=StandardScaler(), pca_thresh=None, do_importances=True):
    """ perform cross validation """
    return decode(X_train=X_train, y_train=y_train, estimator=estimator,
                  n_runs=n_splits, shuffle_splits=shuffle_splits, X_test=None,
                  y_test=None, scaler=scaler, pca_thresh=pca_thresh,
                  do_importances=do_importances)


def final_evaluate(X_train, y_train, estimator, n_runs, X_test=None,
                   y_test=None, scaler=StandardScaler(), pca_thresh=None,
                   do_importances=True):
    """ perform final evaluation """
    return decode(X_train=X_train, y_train=y_train, estimator=estimator,
                  n_runs=n_runs, shuffle_splits=False, X_test=X_test,
                  y_test=y_test, scaler=scaler, pca_thresh=pca_thresh,
                  do_importances=do_importances)


def get_groups_from_cropped(X):
    """ calculate groups from cropped signals. used to average predictions """
    groups = []
    for trial_i, trial_features in enumerate(X):
        groups.extend(len(trial_features) * [trial_i])
    return groups


def decode(X_train, y_train, estimator, n_runs, shuffle_splits, X_test=None,
           y_test=None, scaler=StandardScaler(), pca_thresh=None,
           do_importances=True):
    """
    Perform cross-validation or final evaluation.

    Parameters
    ----------
    X_train: list
        list of data frames with n_windows x n_features
    y_train: list
        train targets
    estimator:
        an estimator following sickit-learn api
    n_runs: int
        number of cv splits or final evaluation repetitions
    shuffle_splits: bool
        whether to shuffle cv splits. ignored when X_test, y_test are given
    X_test: list, None
        list of data frames with n_windows x n_features
    y_test: list, None
        test targets
    scaler:
        feature transformer following scikit-learn api
    pca_thresh: float, int
        threshold for application of principal component analysis. can be float
        for percentage of explained variance to keep or int for number of
        components
    do_importances: bool
        whether or not to estimate feature importances

    Returns dict{set_name: predictions}, dict{setname: additional info}
    -------
    """
    cv_or_eval = "valid" if X_test is None and y_test is None else "eval"
    groups = get_groups_from_cropped(X_train)

    if cv_or_eval == "eval":
        X, y, _, _, train_groups, _ = get_train_test(
            X_train, y_train, np.arange(len(X_train)), [], groups)
        test_groups = get_groups_from_cropped(X_test)
        X_test, y_test, _, _, test_groups, _ = get_train_test(
            X_test, y_test, np.arange(len(X_test)), [], test_groups)
    else:
        kf = KFold(n_splits=n_runs, shuffle=shuffle_splits)

    set_preds = {cv_or_eval: pd.DataFrame(),
                 "train": pd.DataFrame()}
    set_info = {"feature_importances": pd.DataFrame(),
                "rfpimp_importances": pd.DataFrame(),
                "pca_components": pd.DataFrame(),
                "random_states": pd.DataFrame()}

    for run_i in range(n_runs):
        logging.debug("this is run {}".format(run_i))
        if hasattr(estimator, "random_state"):
            estimator.random_state = run_i
            logging.debug("set random state to {}".format(run_i))

        if cv_or_eval == "valid":
            # generator cannot be indexed
            splits = kf.split(np.unique(groups))
            for i, (train_ind, test_ind) in enumerate(splits):
                if i == run_i:
                    break

            X, y, X_test, y_test, train_groups, test_groups = \
                get_train_test(X_train, y_train, train_ind, test_ind, groups)

        preds_train, preds, dict_of_dfs = decode_once(
            X, X_test, y, y_test, estimator, scaler, pca_thresh, do_importances)
        preds_df = preds_to_df(run_i, preds, y_test, test_groups)
        set_preds[cv_or_eval] = set_preds[cv_or_eval].append(preds_df)
        preds_train_df = preds_to_df(run_i, preds_train, y, train_groups)
        set_preds["train"] = set_preds["train"].append(preds_train_df)

        for key, value in dict_of_dfs.items():
            if key == "pca_components":
                value["id"] = pd.Series([run_i] * len(value), index=value.index)
            set_info[key] = set_info[key].append(value, ignore_index=True)

    keys = list(set_info.keys())
    for key in keys:
        if set_info[key].size == 0:
            set_info.pop(key)

    return set_preds, {cv_or_eval: set_info}
