from sklearn.ensemble import RandomForestClassifier
import numpy as np

from brainfeatures.data_set.tuh_abnormal import TuhAbnormalTrain
from brainfeatures.decoding.decode import tune

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(200, 2000, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 7, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]
criterion = ["gini", "entropy"]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}

n_iter = 100
n_jobs = 8
random_state = None
shuffle_splits = False
n_splits = 5
write_header_on_first_run = True
n_recordings = None
n_windows = None
include_default_result = True
agg_func = None
target = "pathological"
out_file = "/data/schirrmr/gemeinl/results/rf/cv/tuning_"+target+"_unagged.csv"
in_dir = "/data/schirrmr/gemeinl/tuh-abnormal-eeg/feats/unagged/full/resampy0.2.1_clipafter/v2.0.0/"

train_feats = TuhAbnormalTrain(in_dir, extension=".npy",
                               n_recordings=n_recordings, target=target,
                               max_recording_mins=None)
train_feats.load()

X, y = [], []
for i, (x, sfreq, label) in enumerate(train_feats):
    if agg_func is not None:
        X.append(agg_func(x, axis=0))
    else:
        if n_windows is not None:
            X.append(x[:n_windows])
        else:
            X.append(x)
    y.append(label)
y = np.array(y)

rf = RandomForestClassifier()
res = tune(X, y, rf, random_grid, n_iter=n_iter, n_jobs=n_jobs,
           random_state=random_state, shuffle_splits=shuffle_splits, n_splits=n_splits,
           out_file=out_file, write_header_on_first_run=write_header_on_first_run)
