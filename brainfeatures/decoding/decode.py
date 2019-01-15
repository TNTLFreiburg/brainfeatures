from sklearn.model_selection import KFold, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition.pca import PCA
import pandas as pd
import numpy as np
import logging
# import rfpimp
import time

from brainfeatures.analysis.analyze import analyze_quality_of_predictions


def get_X_y(data_set, agg_f=None):
    """ read all data from the data set """
    X, y = [], []
    for i, (x, sfreq, label) in enumerate(data_set):
        if agg_f is not None and hasattr(x[0][0], "__len__"):
            x = (agg_f(x, axis=0))
        X.append(x)
        y.append(label)
    y = np.array(y)
    return X, y


def get_cropped_train_test(X, y, train_ind, test_ind, epoch_to_group_map):
    """ split cropped data and target wrt given test ind, s.t. no group is
    accidentally split """
    assert len(X) == len(y)
    assert not (set(train_ind) & set(test_ind)), \
        "train set and test set overlap!"
    unique_groups = np.unique(epoch_to_group_map)
    unique_test_groups = np.array(unique_groups)[test_ind]
    assert len(unique_groups) == len(X)
    feature_labels = X[0].columns

    train_groups, test_groups = [], []
    X_train, y_train, X_test, y_test = [], [], [], []
    for trial_i in range(len(X)):
        if trial_i in unique_test_groups:
            X_test.extend(np.array(X[trial_i]))
            y_test.extend([y[trial_i]] * len(X[trial_i]))
            test_groups.extend([trial_i] * len(X[trial_i]))
        else:
            X_train.extend(np.array(X[trial_i]))
            y_train.extend([y[trial_i]] * len(X[trial_i]))
            train_groups.extend([trial_i] * len(X[trial_i]))
    X_test = pd.DataFrame(X_test, columns=feature_labels)
    X_train = pd.DataFrame(X_train, columns=feature_labels)
    return X_train, y_train, X_test, y_test, train_groups, test_groups


def apply_scaler(X_train, X_test, scaler=StandardScaler()):
    """ fit and tranform a train set, transform a test set accordingly """
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
    components = pd.DataFrame(pca.components_, columns=feature_labels, index=rows)
    return X_train, X_test, components


def decode_once(X_train, X_test, y_train, clf, scaler=StandardScaler(),
                pca_thresh=None):
    """ take train and test set, maybe apply a scaler or pca, fit train set,
    predict test set, return predictions"""
    logging.debug("{} examples in train set, {} examples in test set".format(
        len(X_train), len(X_test)))

    feature_labels = list(X_train.columns)

    feature_importances, pca_components = None, None
    if scaler is not None:
        X_train, X_test = apply_scaler(X_train, X_test, scaler)
    if pca_thresh is not None:
        X_train, X_test, pca_components = apply_pca(X_train, X_test, pca_thresh)
    clf = clf.fit(X_train, y_train)
    # TODO: make sure principle components and feature importances are in the same order
    if hasattr(clf, "feature_importances_"):
        if pca_thresh is not None:
            feature_labels = list(pca_components.index)
        # save random forest feature importances for analysis
        feature_importances = pd.DataFrame([clf.feature_importances_],
                                           columns=feature_labels)
        # TODO: use rfpimp package to get more reliable feature importances
        # print("now doing rfpimp importances")
        # rfpimp_importances = rfpimp.permutation_importances(
        #     clf, X_train, y_train, accuracy_score)
        # info.update({"rfpimp_feature_importances": rfpimp_importances})

    if hasattr(clf, "predict_proba"):
        # save probabilities of positive class (equal to 1 - negaive class)
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
    return y_hat_train, y_hat, feature_importances, pca_components


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


# TODO: move this to optim module?
def tune_generator(X, y, clf, random_grid, n_iter, n_splits=5, shuffle_splits=False,
         random_state=42, n_jobs=1):
    sampler = ParameterSampler(param_distributions=random_grid, n_iter=n_iter,
                               random_state=random_state)
    for i, conf in enumerate(sampler):
        logging.debug("this is run {}".format(i))
        # run with multiple cpus if possible
        if hasattr(clf, "n_jobs"):
            conf.update({"n_jobs": n_jobs})
        clf.__init__(**conf)

        # train and evaluate the classifier
        start = time.time()
        cv_results, cv_info = validate(X, y, clf, n_splits=n_splits,
                                       shuffle_splits=shuffle_splits,
                                       scaler=StandardScaler(), pca_thresh=None)
        mean_metrics = analyze_quality_of_predictions(
            cv_results["predictions"]).mean()
        end = time.time()

        # update config with results and yield
        for metric in mean_metrics.index:
            conf.update({metric: mean_metrics[metric],
                         "n_splits": n_splits,
                         "shuffle_splits": shuffle_splits,
                         "classifier": clf.__class__.__name__,
                         "runtime": end - start,
                         "examples": len(X)})
        yield conf


# TODO: move this to optim module?
def write_tune_result_successively_to_csv(out_file, res, write_header=False):
    if out_file is not None:
        with open(out_file, 'a') as csv_file:
            res.tail(1).to_csv(
                csv_file, header=write_header)


# TODO: move this to optim module?
# TODO: include default result?
def tune(X, y, clf, random_grid, n_iter, n_splits=5, shuffle_splits=False,
         random_state=42, n_jobs=1, out_file=None, write_header_on_first_run=True):
    tune_gen = tune_generator(X=X, y=y, clf=clf, random_grid=random_grid,
                              n_splits=n_splits, shuffle_splits=shuffle_splits,
                              n_iter=n_iter, n_jobs=n_jobs,
                              random_state=random_state)

    res = pd.DataFrame()
    # for comparison, add default clf config result
    # if include_default_result:
    #     default_result, default_info = validate(X, y, clf, n_splits, shuffle_splits)
    #     res = res.append(default_result, ignore_index=True)
    #     write_tune_result_successively_to_csv(
    #         out_file, res, write_header_on_first_run)

    # save all runs successively to a file / to a data frame
    for tune_i, tune_result in enumerate(tune_gen):
        res = res.append(tune_result, ignore_index=True)
        write_tune_result_successively_to_csv(
            out_file, res, tune_i == 0 and write_header_on_first_run)
    return res


# TODO: merge with final evaluate?
def validate(X, y, clf, n_splits, shuffle_splits,
             scaler=StandardScaler(), pca_thresh=None):
    """ do special cross-validation: split data in n_splits, evaluate
     model on every test fold using a different seed (=fold_id)"""
    predictions_by_fold = pd.DataFrame()
    train_predictions_by_fold = pd.DataFrame()
    all_feature_importances = pd.DataFrame()
    all_pca_components = pd.DataFrame()
    info = {}

    groups = []
    for trial_i, trial_features in enumerate(X):
        groups.extend(len(trial_features) * [trial_i])

    # when doing cross-validation do not repeat but run folds with different
    # seeds
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

        predictions_train, predictions, feature_importances, pca_components = \
            decode_once(X_train, X_test, y_train, clf, scaler, pca_thresh)

        predictions_df = create_df_from_predictions(
            fold_id, predictions, y_test, test_groups)
        predictions_by_fold = predictions_by_fold.append(predictions_df)

        train_predictions_df = create_df_from_predictions(
            fold_id, predictions_train, y_train, train_groups)
        train_predictions_by_fold = train_predictions_by_fold.append(
            train_predictions_df)

        if feature_importances is not None:
            all_feature_importances = all_feature_importances.append(
                feature_importances, ignore_index=True)
        # move id to first column?
        if pca_components is not None:
            pca_components["id"] = pd.Series([fold_id] * len(pca_components),
                                             index=pca_components.index)
            all_pca_components = all_pca_components.append(
                pca_components, ignore_index=True)

    if all_feature_importances.size > 0:
        info.update({"feature_importances": all_feature_importances})
    if all_pca_components.size > 0:
        info.update({"pca_components": all_pca_components})

    return {"valid": predictions_by_fold,
            "train": train_predictions_by_fold}, {"valid": info}


# TODO: merge with validate?
# TODO: set random state?
def final_evaluate(X, y, X_eval, y_eval, clf, n_repetitions,
                   scaler=StandardScaler(), pca_thresh=None):
    """ do final evaluation on held-back evaluation set. this should only be
    done once """
    predictions_by_repetition = pd.DataFrame()
    all_feature_importances = pd.DataFrame()
    all_pca_components = pd.DataFrame()
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

        predictions_train, predictions, feature_importances, pca_components = \
            decode_once(X, X_eval, y, clf, scaler, pca_thresh)
        predictions_df = create_df_from_predictions(
            repetition_id, predictions, y_eval, eval_groups)
        predictions_by_repetition = predictions_by_repetition.append(
            predictions_df)

        if feature_importances is not None:
            all_feature_importances = all_feature_importances.append(
                feature_importances, ignore_index=True)
        # move id to first column?
        if pca_components is not None:
            pca_components["id"] = pd.Series([repetition_id] * len(pca_components),
                                             index=pca_components.index)
            all_pca_components = all_pca_components.append(
                pca_components, ignore_index=True)

    if all_feature_importances.size > 0:
        info.update({"feature_importances": all_feature_importances})
    if all_pca_components.size > 0:
        info.update({"pca_components": all_pca_components})

    return {"eval": predictions_by_repetition}, \
           {"eval": {"feature_importances": all_feature_importances}}


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
