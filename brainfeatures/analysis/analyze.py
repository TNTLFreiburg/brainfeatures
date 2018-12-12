from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import logging


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

