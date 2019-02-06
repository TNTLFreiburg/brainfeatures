import logging

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

from brainfeatures.visualization.visualize import plot_feature_correlations, \
    plot_mean_feature_importances_spatial, plot_scaled_mean_importances


# when using someting like 'decision_function' predictions are not within 0, 1 range
# hence, scale them
def scale_predictions_to_0_1(y_pred):
    # TODO: how to know when to use this? don't use for regression,
    # TODO: use for everything else?
    return minmax_scale(y_pred)


# def accuracy_score(y_true, y_pred):
#     if not np.unique(y_true) == np.unique(y_pred) == 2:
#         raise ValueError
#     return np.mean(y_true == y_pred)


def apply_metric(y_true, y_pred, metric):
    try:
        performance = metric(y_true, y_pred)
    except ValueError:
        y_hat = labels_from_continuous(y_pred)
        try:
            performance = metric(y_true, y_hat)
        except ValueError:
            performance = None
    return performance


def labels_from_continuous(y_pred, thresh=.5):
    y_pred = (y_pred >= thresh).astype(int)
    return y_pred


# make sure predictions are not shuffled!?
# should not be a problem anymore since they are organized in groups
# groups are not split in cv
def analyze_quality_of_predictions(predictions, metrics=accuracy_score):
    """ for given predictions, compute given metrics """
    if not hasattr(metrics, "__len__"):
        metrics = [metrics]

    df = pd.DataFrame()
    # id is for the repetition / fold
    for i, id_df in predictions.groupby("id"):
        id_df = id_df.drop("id", axis=1)
        y_true = id_df.y_true
        y_pred = id_df.y_pred

        # group is for the number of crops in a trial
        # average crop predictions
        mean_y_true = id_df.groupby("group").mean().y_true
        mean_y_pred = id_df.groupby("group").mean().y_pred

        row = {}
        for metric in metrics:
            metric_name = metric.__name__
            if len(id_df[id_df.group == 0]) > 1:
                performance = apply_metric(mean_y_true, mean_y_pred, metric)
                row.update({metric_name: performance})
                metric_name = 'crop_' + metric_name

            performance = apply_metric(y_true, y_pred, metric)
            row.update({metric_name: performance})

        df = df.append(row, ignore_index=True)
    return df


def analyze_feature_correlations(feature_matrices):
    """ analyze feature correlations """
    # TODO: add outer correlation?
    # TODO: compare to rfpimp correlation map?
    # check whether feature matrices are 2d or 3d
    # compute inner / outer correlations
    # compute correlations by class
    crop_crounts = [len(d) for d in feature_matrices]
    feature_matrices = pd.concat(feature_matrices, axis=0, ignore_index=True)
    feature_labels = feature_matrices.columns
    feature_labels_without_meta = [label for label in feature_labels
                                   if not label.startswith("meta_")]

    # average over crops
    groups = []
    for i in range(len(crop_crounts)):
        groups.extend(crop_crounts[i] * [i])
    assert len(groups) == len(feature_matrices)
    feature_matrices["group"] = pd.Series(groups)
    feature_matrices = feature_matrices.groupby("group").mean()

    # compute feature correlations
    correlations, pvalues = spearmanr(feature_matrices)

    domains = [label.split("_")[0] for label in feature_labels_without_meta]
    if "meta" in domains:
        domains.remove("meta")
    counts = [domains.count(domain) for domain in np.unique(domains)]
    ticks_at = np.cumsum([0] + counts)
    ticks_at2 = np.cumsum(counts)
    plot_feature_correlations(correlations, xticks=ticks_at[:-1],
                              xticklabels=feature_labels_without_meta,
                              yticks=ticks_at2,
                              yticklabels=feature_labels_without_meta)
    return pd.DataFrame(correlations, columns=feature_labels_without_meta)


def analyze_pca_components(pca_components):
    """ show the features that explain most variance in principle components """
    # TODO: test / check / make sure that order is not changed by pca!
    feature_labels = pca_components.columns
    max_variance_features = []
    for i, g in pca_components.groupby("id"):
        g = g.drop("id", axis=1)
        d = np.argmax(np.abs(g.values), axis=1)
        max_variance_features.append(list(feature_labels[d]))
    return max_variance_features


def analyze_feature_importances(feature_importances):
    """ analyze importance of features (as returned by rf) """
    # visualize top 5 important features per electrode on head scheme
    # average over individual features / electrodes / frequency bands
    feature_labels = feature_importances.columns
    mean_importances = np.mean(feature_importances, axis=0)
    plot_mean_feature_importances_spatial(mean_importances,
                                          feature_labels)

    # average over domains / electrodes / freq bands and plot a bar chart
    domains = [feature_label.split("_")[0]
               for feature_label in feature_labels]
    domains = np.unique(domains)

    electrodes = [feature_label.split("_")[-1]
                  for feature_label in feature_labels
                  if len(feature_label.split("_")) > 2]
    electrodes = np.unique(electrodes)
    electrodes = [electrode for electrode in electrodes
                  if "-" not in electrode]

    freq_bands = [feature_label.split("_")[-2]
                  for feature_label in feature_labels
                  if len(feature_label.split("_")) == 4]
    freq_bands = [freq_band for freq_band in freq_bands if "-" in freq_band]
    freq_bands, indices = np.unique(freq_bands, return_index=True)
    indices, freq_bands = zip(*sorted(zip(indices, freq_bands)))

    plot_scaled_mean_importances([(domains, "domain")],
                                 mean_importances, feature_labels)
    plot_scaled_mean_importances([(freq_bands, "freq bands")],
                                 mean_importances, feature_labels)
    plot_scaled_mean_importances([(electrodes, "electrode")],
                                 mean_importances, feature_labels)


def analyze(predictions, feature_matrices=None, feature_importances=None,
            labels=None):
    """ """
    logging.info("analyzing quality of predictions")
    analyze_quality_of_predictions(predictions)
    feature_labels = feature_matrices.columns
    if feature_matrices is not None and feature_labels is not None and \
            labels is not None:
        logging.info("analyzing feature correlations")
        analyze_feature_correlations(feature_matrices)
    if feature_importances is not None and feature_labels is not None:
        logging.info("analyzing feature importances")
        analyze_feature_importances(feature_importances)
