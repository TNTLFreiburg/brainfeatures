import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition.pca import PCA
import pandas as pd
import numpy as np
import rfpimp


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
    X_grouped, y_grouped, groups = [], [], []
    for trial_i in unique_groups:
        X_grouped.extend(np.array(X[trial_i]))
        y_grouped.extend([y[trial_i]] * len(X[trial_i]))
        groups.extend([trial_i] * len(X[trial_i]))
    X_grouped = pd.DataFrame(X_grouped, columns=feature_labels)
    return X_grouped, y_grouped, groups


def get_train_test(X, y, train_ind, test_ind, epoch_to_group_map):
    """ split cropped data and target wrt given test ind, s.t. no group is
    accidentally split

    X : list of 2-dim pandas df
    y: list of targets
    """
    assert len(X) == len(y), "number of examples and labels does not match"
    assert not (set(train_ind) & set(test_ind)), "train and test set overlap!"
    if hasattr(X[0], "columns"):
        feature_labels = X[0].columns
    else:
        feature_labels = [str(i) for i in range(0, X[0].shape[-1])]

    unique_groups = []
    for group in epoch_to_group_map:
        if group not in unique_groups:
            unique_groups.append(group)
    unique_groups = np.array(unique_groups)

    unique_test_groups = unique_groups[test_ind]
    unique_train_groups = unique_groups[train_ind]
    assert len(unique_groups) == len(X)

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
    components = pd.DataFrame(pca.components_, columns=feature_labels,
                              index=rows)
    return X_train, X_test, components


def decode_once(X_train, X_test, y_train, y_test, clf, scaler=StandardScaler(),
                pca_thresh=None, do_importances=True):
    """ take train and test set, maybe apply a scaler or pca, fit train set,
    predict test set, return predictions"""
    logging.debug("{} examples in train set, {} examples in test set".format(
        len(X_train), len(X_test)))

    feature_labels = X_train.columns

    dict_of_dfs = {}
    if scaler is not None:
        X_train, X_test = apply_scaler(X_train, X_test, scaler)
    if pca_thresh is not None:
        X_train, X_test, pca_components = apply_pca(X_train, X_test,
                                                    pca_thresh)
        dict_of_dfs.update({"pca_components": pca_components})
    clf = clf.fit(X_train, y_train)
    # TODO: for svm set probability to True?
    if do_importances:
        if pca_thresh is not None and hasattr(clf, "feature_importances_"):
            feature_labels = list(pca_components.index)
            # save random forest feature importances for analysis
            feature_importances = pd.DataFrame(
                [clf.feature_importances_], columns=feature_labels)
            dict_of_dfs.update({"feature_importances": feature_importances})

        # rfpimp performances can be applied to any scikit-learn model!
        # rfpimp wants everything as data frame. make sure it gets it
        rfpimp_importances = rfpimp.importances(
            clf, pd.DataFrame(X_test),
            pd.DataFrame(y_test), sort=False)
        dict_of_dfs.update({"rfpimp_importances": rfpimp_importances.T})

    if hasattr(clf, "predict_proba"):
        # save probabilities of positive class (equal to 1 - negative class)
        y_hat_train = clf.predict_proba(X_train)
        y_hat_train = y_hat_train[:, -1]
        y_hat = clf.predict_proba(X_test)
        y_hat = y_hat[:, -1]
    elif hasattr(clf, "decision_function"):
        # for auc save svm distance of points to margin
        y_hat = clf.decision_function(X_test)
        y_hat_train = clf.decision_function(X_train)
    else:
        # create labels
        y_hat = clf.predict(X_test)
        y_hat_train = clf.predict(X_train)
    return y_hat_train, y_hat, dict_of_dfs


def create_df_from_predictions(id_, predictions, y_true, groups=None):
    """ create a pandas data frame from predictions and labels to an id"""
    predictions_df = pd.DataFrame()
    if groups is not None:
        assert len(groups) == len(y_true), \
            "groups: {}, y_true: {}".format(len(groups), len(y_true))

    for i in range(len(y_true)):
        row = {"id": id_,
               "y_true": y_true[i],
               "y_pred": predictions[i]}
        if groups is not None:
            row.update({"group": groups[i]})
        predictions_df = predictions_df.append(row, ignore_index=True)
    return predictions_df


def final_cross_evalidate(X_train, y_train, clf, n_runs, shuffle_splits,
                          X_test=None, y_test=None, scaler=StandardScaler(),
                          pca_thresh=None, do_importances=True):
    """
    Perform cross-validation or final evaluation.

    Parameters
    ----------
    X_train: list
        list of data frames with n_windows x n_features
    y_train: list
        train targets
    clf:
        an estimator following sickit-learn api
    n_runs: int
        number of cv splits or final evaluation repetitions
    shuffle_splits: bool
        whether to shuffle cv splits. ignored when X_test, y_test are given
    X_test: list
        list of data frames with n_windows x n_features
    y_test: list
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
    if X_test is not None and y_test is not None:
        cv_or_eval = "eval"
    else:
        cv_or_eval = "valid"
    all_feature_imps = pd.DataFrame()
    all_rfpimp_imps = pd.DataFrame()
    all_pca_pcs = pd.DataFrame()
    all_predictions = pd.DataFrame()
    info = {}

    groups = []
    for trial_i, trial_features in enumerate(X_train):
        groups.extend(len(trial_features) * [trial_i])

    if cv_or_eval == "eval":
        X, y, _, _, _, _ = get_train_test(X_train, y_train,
                                          np.arange(len(X_train)), [], groups)
        test_groups = []
        for trial_i, trial_features in enumerate(X_test):
            test_groups.extend(len(trial_features) * [trial_i])
        X_test, y_test, _, _, test_groups, _ = get_train_test(
            X_test, y_test, np.arange(len(X_test)), [], test_groups)
    else:
        kf = KFold(n_splits=n_runs, shuffle=shuffle_splits)
        splits = kf.split(np.unique(groups))

    for run_i in range(n_runs):
        logging.debug("this is run {}".format(run_i))
        if cv_or_eval == "valid":
            if hasattr(clf, "random_state"):
                clf.random_state = run_i
                logging.debug("set random state to {}".format(run_i))

            for i, (train_ind, test_ind) in enumerate(splits):
                if i == run_i:
                    break

            X, y, X_test, y_test, train_groups, test_groups = \
                get_train_test(X_train, y_train, train_ind, test_ind, groups)

        predictions_train, predictions, dict_of_dfs = decode_once(
            X, X_test, y, y_test, clf, scaler, pca_thresh, do_importances)
        predictions_df = create_df_from_predictions(run_i, predictions,
                                                    y_test, test_groups)
        all_predictions = all_predictions.append(predictions_df)

        for key, value in dict_of_dfs.items():
            if key == "feature_importances":
                all_feature_imps = all_feature_imps.append(value,
                                                           ignore_index=True)
            elif key == "rfpimp_importances":
                all_rfpimp_imps = all_rfpimp_imps.append(value,
                                                         ignore_index=True)
            elif key == "pca_components":
                value["id"] = pd.Series([run_i] * len(value),
                                        index=value.index)
                all_pca_pcs = all_pca_pcs.append(value, ignore_index=True)

    if all_feature_imps.size > 0:
        info.update({"feature_importances": all_feature_imps})
    if all_rfpimp_imps.size > 0:
        info.update({"rfpimp_importances": all_rfpimp_imps})
    if all_pca_pcs.size > 0:
        info.update({"pca_components": all_pca_pcs})

    return {cv_or_eval: all_predictions}, \
           {cv_or_eval: info}


def decode(train_set, clf, n_splits_or_repetitions, shuffle_splits,
           eval_set=None, scaler=StandardScaler(), pca_thresh=None):
    """ decode the target from given data sets, either doing validation or
    final evaluation """
    X_test, y_test = None, None
    if eval_set is not None:
        X_test, y_test = get_X_y(eval_set)

    X, y = get_X_y(train_set)
    results, info = final_cross_evalidate(X, y, clf, n_splits_or_repetitions,
                                          shuffle_splits, X_test=X_test,
                                          y_test=y_test, scaler=scaler,
                                          pca_thresh=pca_thresh)
    return results, info
