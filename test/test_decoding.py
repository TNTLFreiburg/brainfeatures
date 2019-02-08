from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from brainfeatures.decoding.decode import (validate, final_evaluate,
                                           get_train_test, decode_once,
                                           apply_pca, apply_scaler)


def test_validate():
    seed = 0
    np.random.seed(seed)
    clf = RandomForestClassifier(random_state=seed)

    X = np.random.rand(10*10).reshape(10, 10)
    y = np.array(5 * [0] + 5 * [1])
    train_ind = [0, 1, 2, 3, 4]
    test_ind = [5, 6, 7, 8, 9]
    clf.fit(X[train_ind], y[train_ind])
    probas1 = clf.predict_proba(X[test_ind])
    expected_1 = np.array(probas1)[:, -1]

    clf.fit(X[test_ind], y[test_ind])
    probas2 = clf.predict_proba(X[train_ind])
    expected_2 = np.array(probas2)[:, -1]

    X = X[:, np.newaxis]
    result, info = validate(X, y, clf, 2, shuffle_splits=False,
                            scaler=None, pca_thresh=None,
                            do_importances=True)
    preds = result["valid"]
    y_pred = preds[preds.id == 0]["y_pred"]
    np.testing.assert_array_equal(y_pred, expected_1)

    y_pred = preds[preds.id == 1]["y_pred"]
    np.testing.assert_array_equal(y_pred, expected_2)

    assert "feature_importances" in info["valid"]
    assert "rfpimp_importances" in info["valid"]


def test_apply_scaler():
    X_train = np.random.rand(10*10).reshape(10, 10)
    X_test = np.random.rand(10*10).reshape(10, 10)
    X_train = pd.DataFrame(X_train, columns=["a", "b", "c", "d", "e",
                                             "f", "g", "h", "i", "j"])
    X_test = pd.DataFrame(X_test, columns=["a", "b", "c", "d", "e",
                                           "f", "g", "h", "i", "j"])
    X_train_scaled, X_test_scaled = apply_scaler(X_train, X_test)
    np.testing.assert_allclose(np.array(X_train_scaled).mean(), [0],
                               rtol=1e-07, atol=1e-07)
    np.testing.assert_allclose(np.array(X_train_scaled).var(), [1],
                               rtol=1e-07, atol=1e-07)


def test_apply_pca():
    pca_thresh = 3
    X_train = np.random.rand(10*10).reshape(10, 10)
    X_test = np.random.rand(10*10).reshape(10, 10)
    X_train = pd.DataFrame(X_train, columns=["a", "b", "c", "d", "e",
                                             "f", "g", "h", "i", "j"])
    X_test = pd.DataFrame(X_test, columns=["a", "b", "c", "d", "e",
                                           "f", "g", "h", "i", "j"])
    X_train_pc, X_test_pc, components = apply_pca(X_train, X_test, pca_thresh)
    assert len(components) == pca_thresh
    assert X_train_pc.shape[-1] == pca_thresh
    assert X_test_pc.shape[-1] == pca_thresh


# def test_get_train_test():
#     # 10 examples, 1 window, 10 features
#     X = np.random.rand(10*10).reshape(10, 10)
#     y = np.array(5 * [0] + 5 * [1])
#     train_ind = [0, 3, 1, 2, 4, 7, 6, 5]
#     test_ind = [9, 8]
#
#     X_train, y_train, X_test, y_test = \
#         get_train_test(X, y, train_ind, test_ind, np.arange(10))
#     np.testing.assert_array_equal(X[train_ind], X_train)
#     np.testing.assert_array_equal(y[train_ind], y_train)
#     np.testing.assert_array_equal(X[test_ind], X_test)
#     np.testing.assert_array_equal(y[test_ind], y_test)
#     assert y_train[4] == 0
#     assert y_train[-1] == 1
#     # in this case we are not actually doing cropped decoding
#     # every trial is in its own group
#     X = X[:, np.newaxis, :]
#     epoch_to_group_map = []
#     for i, x in enumerate(X):
#         epoch_to_group_map.extend(len(x) * [i])
#     assert len(epoch_to_group_map) == len(X)
#
#     X_train, y_train, X_test, y_test, train_groups, test_groups = \
#         get_train_test(X, y, train_ind, test_ind, epoch_to_group_map)
#     np.testing.assert_array_equal(X[train_ind].squeeze(), X_train)
#     np.testing.assert_array_equal(X[test_ind].squeeze(), X_test)
#     np.testing.assert_array_equal(y[train_ind], y_train)
#     np.testing.assert_array_equal(y[test_ind], y_test)
#     assert y_train[4] == 0
#     assert y_train[-1] == 1


def test_get_cropped_train_test():
    # 10 examples, 9 windows, 8 features
    X = np.random.rand(10*9*8).reshape(10, 9, 8)
    epoch_to_group_map = []
    for i, x in enumerate(X):
        epoch_to_group_map.extend(len(x) * [i])
    y = np.array(7 * [0] + 3 * [1])
    train_ind = [0, 1, 2, 3, 4, 5, 6, 7]
    test_ind = [8, 9]
    X_train, y_train, X_test, y_test, train_groups, test_groups = \
        get_train_test(X, y, train_ind, test_ind, epoch_to_group_map)
    expected = len(X[-2]) * [test_ind[-2]] + len(X[-1]) * [test_ind[-1]]
    np.testing.assert_array_equal(expected, test_groups)
    assert len(y_train) == 8 * 9
    assert y_train[7*9] == 1
    assert y_train[7*9-1] == 0
    assert len(X_test.shape) == 2


def test_decode_once():
    seed = 0
    np.random.seed(seed)
    clf = RandomForestClassifier(random_state=seed)

    X = np.random.rand(10*10).reshape(10, 10)
    y = np.array(5 * [0] + 5 * [1])
    train_ind = [0, 3, 1, 2, 4, 7, 6, 5]
    test_ind = [9, 8]
    clf.fit(X[train_ind], y[train_ind])
    probas = clf.predict_proba(X[test_ind])
    expected = probas[:, 1]

    X_train = pd.DataFrame(X[train_ind], columns=["a", "b", "c", "d", "e",
                                                  "f", "g", "h", "i", "j"])
    X_test = pd.DataFrame(X[test_ind], columns=["a", "b", "c", "d", "e",
                                                "f", "g", "h", "i", "j"])
    result_train, result, info = decode_once(
        X_train, X_test, y[train_ind], y[test_ind],
        clf, scaler=None, pca_thresh=None, do_importances=False)
    np.testing.assert_array_equal(expected, result)
    assert "feature_importances" not in info
