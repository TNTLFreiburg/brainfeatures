import numpy as np
from sklearn.ensemble import RandomForestClassifier

from brainfeatures.decoding.decode import validate, final_evaluate, \
    get_train_test, get_cropped_train_test, decode_once, apply_pca, apply_scaler


def test_validate():
    seed = 0
    np.random.seed(seed)
    clf = RandomForestClassifier(random_state=seed)

    X = np.random.rand(10*10).reshape(10, 10)
    y = np.array(5 * [0] + 5 * [1])
    train_ind = [0, 1, 2, 3, 4]
    test_ind = [5, 6, 7, 8, 9]
    clf.fit(X[train_ind], y[train_ind])
    probas = clf.predict_proba(X[test_ind])
    expected = np.array(probas)[:, -1]

    result, info = validate(X, y, clf, 2, False, None, None)
    preds = result["predictions"]
    y_pred = preds[preds.id == 0]["y_pred"]
    np.testing.assert_array_equal(y_pred, expected)

    clf.fit(X[test_ind], y[test_ind])
    probas = clf.predict_proba(X[train_ind])
    expected = np.array(probas)[:, -1]

    y_pred = preds[preds.id == 1]["y_pred"]
    np.testing.assert_array_equal(y_pred, expected)

    assert "feature_importances" in info[0]


# def test_validate_cropped():
#     pass


def test_apply_scaler():
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = np.random.rand(10*10).reshape(10, 10)
    X_test = np.random.rand(10*10).reshape(10, 10)
    result1, result2 = apply_scaler(X_train, X_test, scaler)
    np.testing.assert_allclose(np.mean(result1), [0], rtol=1e-07, atol=1e-07)
    np.testing.assert_allclose(np.var(result1), [1], rtol=1e-07, atol=1e-07)


def test_apply_pca():
    pca_thresh = .9
    X_train = np.random.rand(10*10).reshape(10, 10)
    X_test = np.random.rand(10*10).reshape(10, 10)
    result1, result2, dict_with_pca = apply_pca(X_train, X_test, pca_thresh)
    # TODO: what to test here?


def test_get_train_test():
    X = np.random.rand(10*10).reshape(10, 10)
    y = np.array(5 * [0] + 5 * [1])
    train_ind = [0, 3, 1, 2, 4, 7, 6, 5]
    test_ind = [9, 8]
    X_train, y_train, X_test, y_test = get_train_test(X, y, train_ind, test_ind)
    np.testing.assert_array_equal(X[train_ind], X_train)
    np.testing.assert_array_equal(y[train_ind], y_train)
    np.testing.assert_array_equal(X[test_ind], X_test)
    np.testing.assert_array_equal(y[test_ind], y_test)
    assert y_train[4] == 0
    assert y_train[-1] == 1


def test_get_cropped_train_test():
    X = np.random.rand(10*9*8).reshape(10, 9, 8)
    epoch_to_group_map = []
    for i, x in enumerate(X):
        epoch_to_group_map.extend(len(x) * [i])
    y = np.array(7 * [0] + 3 * [1])
    train_ind = [0, 1, 2, 3, 4, 5, 6, 7]
    test_ind = [8, 9]
    X_train, y_train, X_test, y_test, test_groups = get_cropped_train_test(
        X, y, train_ind, test_ind, epoch_to_group_map)
    expected = len(X[-2]) * [test_ind[-2]] + len(X[-1]) * [test_ind[-1]]
    np.testing.assert_array_equal(expected, test_groups)
    assert len(y_train) == 8 * 9
    assert y_train[7*9] == 1
    assert y_train[7*9-1] == 0


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

    result, info = decode_once(X[train_ind], X[test_ind], y[train_ind], clf, scaler=None, pca_thresh=None)
    np.testing.assert_array_equal(expected, result)
    assert "feature_importances" in info
