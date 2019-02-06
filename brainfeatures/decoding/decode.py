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


def get_train_test(X, y, train_ind, test_ind):
    """ split data and target wrt given train and test indeces """
    assert not (set(train_ind) & set(test_ind)), "train and test set overlap!"
    X = np.array(X)
    y = np.array(y)
    X_train = X[train_ind]
    y_train = y[train_ind]
    X_test = X[test_ind]
    y_test = y[test_ind]
    return X_train, y_train, X_test, y_test


def get_cropped_train_test(X, y, train_ind, test_ind, epoch_to_group_map):
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

    X_test, y_test, test_groups = [], [], []
    for trial_i in unique_test_groups:
        X_test.extend(np.array(X[trial_i]))
        y_test.extend([y[trial_i]] * len(X[trial_i]))
        test_groups.extend([trial_i] * len(X[trial_i]))
    X_test = pd.DataFrame(X_test, columns=feature_labels)

    X_train, y_train, train_groups = [], [], []
    for trial_i in unique_train_groups:
        X_train.extend(np.array(X[trial_i]))
        y_train.extend([y[trial_i]] * len(X[trial_i]))
        train_groups.extend([trial_i] * len(X[trial_i]))
    X_train = pd.DataFrame(X_train, columns=feature_labels)
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
        X_train, X_test, pca_components = apply_pca(X_train, X_test, pca_thresh)
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
        # rfpimp want everything as data frame. make sure it gets it
        rfpimp_importances = rfpimp.importances(
            clf, pd.DataFrame(X_test), pd.DataFrame(y_test), sort=False)
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


# TODO: merge with final evaluate?
def validate(X, y, clf, n_splits, shuffle_splits,
             scaler=StandardScaler(), pca_thresh=None, do_importances=True):
    """ do special cross-validation: split data in n_splits, evaluate
     model on every test fold using a different seed (=fold_id)"""
    feature_importances_by_fold = pd.DataFrame()
    rfpimp_importances_by_fold = pd.DataFrame()
    train_predictions_by_fold = pd.DataFrame()
    pca_components_by_fold = pd.DataFrame()
    predictions_by_fold = pd.DataFrame()
    info = {}

    groups = []
    for trial_i, trial_features in enumerate(X):
        groups.extend(len(trial_features) * [trial_i])

    kf = KFold(n_splits=n_splits, shuffle=shuffle_splits)
    splits = kf.split(np.unique(groups))

    # use yield here?
    for fold_id, (train_ind, test_ind) in enumerate(splits):
        logging.debug("this is fold {}".format(fold_id))
        if hasattr(clf, "random_state"):
            clf.random_state = fold_id
            logging.debug("set random state to {}".format(fold_id))

        X_train, y_train, X_test, y_test, train_groups, test_groups = \
            get_cropped_train_test(X, y, train_ind, test_ind, groups)

        predictions_train, predictions, dict_of_dfs = \
            decode_once(X_train, X_test, y_train, y_test, clf, scaler,
                        pca_thresh, do_importances)

        predictions_df = create_df_from_predictions(
            fold_id, predictions, y_test, test_groups)
        predictions_by_fold = predictions_by_fold.append(predictions_df)

        train_predictions_df = create_df_from_predictions(
            fold_id, predictions_train, y_train, train_groups)
        train_predictions_by_fold = train_predictions_by_fold.append(
            train_predictions_df)

        for key, value in dict_of_dfs.items():
            if key == "feature_importances":
                feature_importances_by_fold = feature_importances_by_fold.append(
                    value, ignore_index=True)
            elif key == "rfpimp_importances":
                rfpimp_importances_by_fold = rfpimp_importances_by_fold.append(
                    value, ignore_index=True)
            elif key == "pca_components":
                value["id"] = pd.Series([fold_id] * len(value), index=value.index)
                pca_components_by_fold = pca_components_by_fold.append(
                    value, ignore_index=True)

    if feature_importances_by_fold.size > 0:
        info.update({"feature_importances": feature_importances_by_fold})
    if rfpimp_importances_by_fold.size > 0:
        info.update({"rfpimp_importances": rfpimp_importances_by_fold})
    if pca_components_by_fold.size > 0:
        info.update({"pca_components": pca_components_by_fold})

    return {"valid": predictions_by_fold,
            "train": train_predictions_by_fold}, {"valid": info}


# TODO: merge with validate?
# TODO: set random state?
def final_evaluate(X, y, X_eval, y_eval, clf, n_repetitions,
                   scaler=StandardScaler(), pca_thresh=None, do_importances=True):
    """ do final evaluation on held-back evaluation set. this should only be
    done once """
    feature_importances_by_rep = pd.DataFrame()
    rfpimp_importances_by_rep = pd.DataFrame()
    pca_components_by_rep = pd.DataFrame()
    predictions_by_rep = pd.DataFrame()
    info = {}

    eval_groups = []
    for trial_i, trial_features in enumerate(X_eval):
        eval_groups.extend(len(trial_features) * [trial_i])
    X_eval, y_eval, _, _, eval_groups, _ = \
        get_cropped_train_test(X_eval, y_eval, np.arange(len(X_eval)), [], eval_groups)
    train_groups = []
    for trial_i, trial_features in enumerate(X):
        train_groups.extend(len(trial_features) * [trial_i])
    X, y, _, _, _, _ = get_cropped_train_test(X, y, np.arange(len(X)), [], train_groups)

    for repetition_id in range(n_repetitions):
        logging.debug("this is repetition {}".format(repetition_id))
        # if hasattr(clf, "random_state"):
        #     clf.random_state = repetition_id
        #     logging.debug("set random state to {}".format(repetition_id))

        predictions_train, predictions, dict_of_dfs = \
            decode_once(X, X_eval, y, y_eval, clf, scaler, pca_thresh,
                        do_importances)

        predictions_df = create_df_from_predictions(
            repetition_id, predictions, y_eval, eval_groups)

        predictions_by_rep = predictions_by_rep.append(
            predictions_df)

        for key, value in dict_of_dfs.items():
            if key == "feature_importances":
                feature_importances_by_rep = feature_importances_by_rep.append(
                    value, ignore_index=True)
            elif key == "rfpimp_importances":
                rfpimp_importances_by_rep = rfpimp_importances_by_rep.append(
                    value, ignore_index=True)
            elif key == "pca_components":
                value["id"] = pd.Series([repetition_id] * len(value), index=value.index)
                pca_components_by_rep = pca_components_by_rep.append(
                    value, ignore_index=True)

    if feature_importances_by_rep.size > 0:
        info.update({"feature_importances": feature_importances_by_rep})
    if rfpimp_importances_by_rep.size > 0:
        info.update({"rfpimp_importances": feature_importances_by_rep})
    if pca_components_by_rep.size > 0:
        info.update({"pca_components": pca_components_by_rep})

    return {"eval": predictions_by_rep}, \
           {"eval": info}


def decode(train_set, clf, n_splits_or_repetitions, shuffle_splits,
           eval_set=None, scaler=StandardScaler(), pca_thresh=None):
    """ decode the target from given data sets, either doing validation or
    final evaluation """
    if eval_set is None:
        logging.info("doing (cross-) validation")
        X, y = get_X_y(train_set)
        results, info = validate(X, y, clf, n_splits_or_repetitions,
                                 shuffle_splits, scaler, pca_thresh)
    else:
        logging.info("doing final evaluation")
        X, y = get_X_y(train_set)
        X_eval, y_eval = get_X_y(eval_set)
        results, info = final_evaluate(X, y, X_eval, y_eval, clf,
                                       n_splits_or_repetitions, scaler,
                                       pca_thresh)
    return results
