from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import logging

from ..visualization.visualize import plot_feature_correlations, \
    plot_mean_feature_importances_spatial, plot_scaled_mean_importances


# when using someting like 'decision_function' predictions are not within 0, 1 range
# hence, scale them
def scale_predictions_to_0_1(y_pred):
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


def labels_from_continuous(y_pred):
    y_pred = (y_pred >= .5).astype(int)
    return y_pred


# make sure predictions are not shuffled!?
# should not be a problem anymore since they are organized in groups
# groups are not split in cv
def analyze_quality_of_predictions(predictions, metrics=accuracy_score):
    """ for given predictions, compute given metrics """
    ids = predictions.id.unique()

    if not hasattr(metrics, "__len__"):
        metrics = [metrics]

    df = pd.DataFrame()
    for id_ in ids:
        # take predictions per fold/repetition
        id_df = predictions[predictions.id == id_]
        y_true = id_df.y_true
        y_pred = id_df.y_pred

        # average crop predictions
        if hasattr(id_df, "group"):
            mean_y_true = id_df.groupby("group").mean().y_true
            mean_y_pred = id_df.groupby("group").mean().y_pred
            # mean_y_pred_labels = [0 if y < thresh else 1 for y in mean_y_pred]

        row = {}
        for metric in metrics:
            metric_name = metric.__name__
            if hasattr(id_df, "group"):
                # performance = metric(mean_y_true, mean_y_pred_labels)
                performance = apply_metric(mean_y_true, mean_y_pred, metric)
                row.update({metric_name: performance})
                metric_name = 'crop_' + metric_name

            performance = apply_metric(y_true, y_pred, metric)
            row.update({metric_name: performance})

        df = df.append(row, ignore_index=True)
    return df


def analyze_feature_correlations(feature_matrices, feature_labels=None,
                                 out_dir=None):
    """ analyze feature correlations """
    # check whether feature matrices are 2d or 3d
    # compute inner / outer correlations
    # compute correlations by class
    # visualize correlation map(s)
    # check if matrix is n_recs x n_windows x n_features
    # or n_recs x n_features
    do_cropped = hasattr(feature_matrices[0][0], "__len__")
    if do_cropped:
        feature_matrices = [np.mean(m, axis=0) for m in feature_matrices]
        feature_matrices = np.array(feature_matrices)
    # compute feature correlations
    correlations, pvalues = spearmanr(feature_matrices)
    # TODO: plot correlation maps of different domains, create ticks and ticklabels
    plot_feature_correlations(correlations, feature_labels)
    return correlations


def analyze_feature_importances(feature_importances, feature_labels, out_dir=None):
    """ analyze importance of features (as returned by rf) """
    # visualize top 5 important features per electrode on head scheme
    # average over individual features / electrodes / frequency bands
    mean_importances = np.mean(feature_importances, axis=0)
    plot_mean_feature_importances_spatial(mean_importances,
                                          feature_labels, out_dir)

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


def analyze_pca_components(pca_components, feature_labels):
    mean_components = np.mean(pca_components, axis=0)
    raise NotImplementedError


def analyze(predictions, feature_matrices=None,
            feature_importances=None, feature_labels=None, labels=None,
            out_dir=None):
    """ """
    logging.info("analyzing quality of predictions")
    analyze_quality_of_predictions(predictions)
    if feature_matrices is not None and feature_labels is not None and \
            labels is not None:
        logging.info("analyzing feature correlations")
        analyze_feature_correlations(feature_matrices, feature_labels, out_dir)
    if feature_importances is not None and feature_labels is not None:
        logging.info("analyzing feature importances")
        analyze_feature_importances(feature_importances, feature_labels,
                                    out_dir)
