from datetime import datetime, date
from collections import OrderedDict
from functools import partial
import logging
import time

from joblib import Parallel, delayed

from brainfeatures.analysis.analyze import (
    analyze_quality_of_predictions, analyze_feature_importances,
    analyze_feature_correlations, analyze_pca_components)
from brainfeatures.decoding.decode import validate, final_evaluate

log = logging.getLogger(__name__)
log.setLevel("INFO")


class Experiment(object):
    def __init__(
            self,
            devel_set,
            analyze_features=True,
            estimator=None,
            eval_set=None,
            feature_generation_function=None,
            feature_generation_params=None,
            feature_vector_modifier=None,
            metrics=None,
            n_jobs=1,
            n_splits_or_repetitions=5,
            pca_thresh=None,
            preproc_function=None,
            preproc_params=None,
            scaler=None,
            shuffle_splits=False,
            verbosity="INFO"):

        self._data_sets = OrderedDict([("devel", devel_set),
                                       ("eval", eval_set)])
        self._feat_gen_params = feature_generation_params
        self._feature_modifier = feature_vector_modifier
        self._feat_gen_f = feature_generation_function
        self._feature_anaylsis = analyze_features
        self._n_runs = n_splits_or_repetitions
        self._preproc_params = preproc_params
        self._shuffle_splits = shuffle_splits
        self._preproc_f = preproc_function
        self._pca_thresh = pca_thresh
        self._estimator = estimator
        self._verbosity = verbosity
        self._metrics = metrics
        self._n_jobs = n_jobs
        self._scaler = scaler

        self._preprocessed = {"devel": [], "eval": []}
        self._features = {"devel": [], "eval": []}
        self._targets = {"devel": [], "eval": []}
        self.info = {"devel": {}, "eval": {}}
        self._feature_names = None
        self.performances = {}
        self.predictions = {}
        self.times = {}

        self._run_checks()
        if feature_generation_params is not None:
            self._feat_gen_f = partial(feature_generation_function,
                                       **feature_generation_params)
        if preproc_params is not None:
            self._preproc_f = partial(preproc_function, **preproc_params)
    """
    Class that performs one feature-based experiment on development (and
    evaluation) set.

    It is structured as follows:

    1. (optional) Read raw data from given data set(s) (devel_set, eval_set) and 
                  apply given preprocessing rules (preproc_function with 
                  preproc_params).
    2. (optional) Take previously preprocessed signals / read preprocessed 
                  signals from given data set(s) (devel_set, eval_set) and 
                  generate features (feature_generation_function with 
                  feature_generation_params).
    3. (optional) Take previously generated features / read generated features
                  from given data set(s) (devel_set, eval_set) and run 
                  (cross)-validation (and final evaluation) using given 
                  classifier (estimator), and evaluate it using given metric(s).

    Parameters
    ----------
    devel_set: :class:`.DataSet`
        with __len__ and __getitem__ returning (example, sfreq, label),
    estimator:  object, optional
        an estimator following scikit-learn api
    metrics: object, list of objects, optional
         metric(s) following scikit-learn api
    eval_set: :class:`.DataSet`
        with __len__ and __getitem__ returning (example, sfreq, label),
        if None, experiment will perform (cross-)validation,
        if not None, experiment will perform final evaluation
    n_jobs: int, optional
        number of jobs to use for parallel cleaning / feature generation
    preproc_function: function, optional
        takes signals and sampling frequency of the signals and returns cleaned
        signals
    preproc_params: dict, optional
        keyword arguments needed for cleaning functions except signals and
        sampling frequency in the form
    feature_generation_function: function, optional
    feature_generation_params: dict, optional
        keyword arguments needed for feature generation functions except
        signals and sampling frequency
    n_splits_or_repetitions: int, optional
        number of (cross-)validation splits / final evaluation repetitions
    shuffle_splits: bool, optional
        shuffles the cross-validation splits
    pca_thresh: integer, float, None, optional
        inter specifying number of components to keep / float specifying
        percentage of explained variance to determine number of components to
        keep with application of principal component analysis
    scaler: object, optional
        a scaler following scikit-learn api used to scale feature values
    verbosity: str, optional
        verbosity level
    """

    def run(self):
        """
        Run complete experiment.
        """
        today, now = date.today(), datetime.time(datetime.now())
        logging.info('Started on {} at {}'.format(today, now))

        preprocess = self._preproc_f is not None
        modify_features = self._feature_modifier is not None
        generate_features = self._feat_gen_f is not None
        predict = self._estimator is not None and self._features["devel"]

        for set_name in self._data_sets.keys():
            if preprocess:
                self._preprocess(set_name)

            if generate_features:
                if not preprocess:
                    self._load(set_name, "preprocessed")
                self._generate_features(set_name)

            if not preprocess and not generate_features:
                self._load(set_name, "features")

            if modify_features:
                self._modify_features(set_name)

        if predict:
            self._decode(set_name)

        today, now = date.today(), datetime.time(datetime.now())
        logging.info("Finished on {} at {}.".format(today, now))

    def _preprocess(self, set_name):
        """
        Apply given reprocessing rules to all examples in data set specified by
        set_name.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        """
        start = time.time()
        logging.info("Preprocessing ({})".format(set_name))
        x_fs_y_pre = Parallel(n_jobs=self._n_jobs)(
            (delayed(self._preproc_f)(x, fs, y)
             for (x, fs, y) in self._data_sets[set_name]))

        for (x_pre, fs_pre, y_pre) in x_fs_y_pre:
            self._preprocessed[set_name].append(x_pre)
            self._targets[set_name].append(y_pre)
        if "sfreq" not in self.info[set_name]:
            self.info[set_name]["sfreq"] = fs_pre
        self.times.setdefault("preprocessing", {}).update(
            {set_name: time.time() - start})

    def _load(self, set_name, set_state):
        """
        Load cleaned signals or features from data set specified by
        set_name.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        set_state: str
            either "preprocessed" or "features"
        """
        start = time.time()
        logging.info("Loading {} ({})".format(set_name, set_state))
        for (x, fs, y) in self._data_sets[set_name]:
            getattr(self, "_" + set_state)[set_name].append(x)
            self._targets[set_name].append(y)
        if "sfreq" not in self.info[set_name]:
            self.info[set_name]["sfreq"] = fs
        if self._feature_names is None:
            self._feature_names = list(x.columns)
        self.times.setdefault("loading", {}).update(
            {set_name: time.time() - start})

    def _modify_features(self, set_name):
        # TODELAY: impove feature vector modifier
        self._features[set_name], self._feature_names = (
            self._feature_modifier(self._data_sets[set_name],
                                   self._features[set_name],
                                   self._feature_names))
        assert len(self._features[set_name]) > 0, (
            "removed all feature vectors")
        assert (self._features[set_name][0].shape[-1] ==
                len(self._feature_names)), (
            "number of features {} and feature names {} does not match"
                .format(self._features[set_name][0].shape[-1],
                        len(self._feature_names)))

    def _generate_features(self, set_name):
        """
        Apply given feature generation procedure to all examples in data set
        specified by set_name.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        """
        start = time.time()
        logging.info("Generating features ({})".format(set_name))
        feature_matrix = Parallel(n_jobs=self._n_jobs)(
            (delayed(self._feat_gen_f)(example, self.info[set_name]["sfreq"])
             for example in self._preprocessed[set_name]))
        i_to_delete = []
        for i, feature_vector in enumerate(feature_matrix):
            if feature_vector is not None:
                self._features[set_name].append(feature_vector)
            # important: if feature generation fails, and therefore feature
            # vector is None, remove according label!
            else:
                i_to_delete.append(i)
        # remove in reverse order to not get confused with indices
        for i in i_to_delete[::-1]:
            del self._targets[set_name][i]
            logging.warning("removed example {} from labels".format(i))
        if self._feature_names is None:
            self._feature_names = list(feature_vector.columns)
        assert len(self._features[set_name]) == len(self._targets[set_name]), (
            "number of feature vectors {} and labels {} does not match".
            format(len(self._features[set_name]), len(self._targets[set_name])))
        self.times.setdefault("feature generation", {}).update(
            {set_name: time.time() - start})

    def _decode(self, set_name):
        """
        Run decoding on data specified by set_name, analyze the performance
        and the features used.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        """
        if set_name == "devel":
            self._validate()
        else:
            assert set_name == "eval", "unknown set name {}".format(set_name)
            self._final_evaluate()

        if self._metrics is not None:
            self._analyze_performance(set_name)

    def _validate(self):
        """
        Perform (cross-)validation on development set.
        """
        start = time.time()
        logging.info("Making predictions (validation)")
        assert len(self._features["devel"]) == len(self._targets["devel"]), (
            "number of feature vectors {} and labels {} does not match"
            .format(len(self._features["devel"]), len(self._targets["devel"])))
        valid_results, valid_info = validate(
            X_train=self._features["devel"],
            y_train=self._targets["devel"],
            estimator=self._estimator,
            n_splits=self._n_runs,
            shuffle_splits=self._shuffle_splits,
            scaler=self._scaler,
            pca_thresh=self._pca_thresh,
            do_importances=self._feature_anaylsis)
        self.predictions.update(valid_results)
        self.info.update(valid_info)
        self.times["validation"] = time.time() - start

    def _analyze_performance(self, set_name):
        """
        Apply specified metrics on predictions on data set specified by
        set_name.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        """
        set_names = ["train", set_name]
        for set_name in set_names:
            if set_name == "devel":
                set_name = "valid"
            logging.info("Computing performances ({})".format(set_name))
            performances = analyze_quality_of_predictions(
                self.predictions[set_name], self._metrics)
            self.performances.update({set_name: performances})
            logging.info("Achieved in average\n{}\n on {} set.".format(
                self.performances[set_name].mean().to_string(), set_name))

    def _final_evaluate(self):
        """
        Perform final evaluation. Train on development and evaluate on final
        evaluation set.
        """
        start = time.time()
        logging.info("Making predictions (final evaluation)")
        assert len(self._features["eval"]) == len(self._targets["eval"]), (
            "number of feature vectors {} and labels {} does not match".
            format(len(self._features["eval"]), len(self._targets["eval"])))
        eval_results, eval_info = final_evaluate(
            X_train=self._features["devel"],
            y_train=self._targets["devel"],
            X_test=self._features["eval"],
            y_test=self._targets["eval"],
            estimator=self._estimator,
            n_runs=self._n_runs,
            scaler=self._scaler,
            pca_thresh=self._pca_thresh,
            do_importances=self._feature_anaylsis)
        self.predictions.update(eval_results)
        self.info.update(eval_info)
        self.times["final evaluation"] = time.time() - start

    def analyze_features(self):
        for set_name in self._data_sets.keys():
            self._analyze_features(set_name)

    def _analyze_features(self, set_name):
        """
        Perform analysis of features specified by set_name using correlation
        coefficient, principle components and feature importances.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        """
        feature_corrs = analyze_feature_correlations(self._features[set_name])
        self.info[set_name].update({"feature_correlations": feature_corrs})
        if set_name == "devel":
            set_name = "valid"
        else:
            assert set_name == "eval", "unknown set name {}".format(set_name)

        if self._pca_thresh is not None:
            pca_features = analyze_pca_components(
                self.info[set_name]["pca_components"])
            self.info[set_name].update({"pca_features": pca_features})

        # do feature importances with principle components?
        if self._pca_thresh is None:
            if "feature_importances" in self.info[set_name]:
                analyze_feature_importances(
                    self.info[set_name]["feature_importances"])

            if "rfpimp_importances" in self.info[set_name]:
                analyze_feature_importances(
                    self.info[set_name]["rfpimp_importances"])

    def _run_checks(self):
        """
        Assure conformity of given arguments.
        """
        # use argparse?
        assert self._verbosity in ["DEBUG", "INFO", "WARNING", "ERROR", 0, 10,
                                   20, 30, 40], ("unknown verbosity level {}"
                                                 .format(self._verbosity))
        if self._feat_gen_f is not None:
            assert hasattr(self._feat_gen_f, "__call__"), (
                "feature_generation_procedure has to be a callable")
        if self._preproc_f is not None:
            assert hasattr(self._preproc_f, "__call__"), (
                "preproc_function has to be a callable")
        if self._preproc_params is not None:
            assert type(self._preproc_params) is dict, (
                "preproc_params has to be a dictionary")
        if self._feat_gen_params is not None:
            assert type(self._feat_gen_params) is dict, (
                "feature_generation_params has to be a dictionary")
        assert len(self._data_sets["devel"][0]) == 3, (
            "__getitem__ of data set needs to return x, fs, y")
        assert hasattr(self._data_sets["devel"][0][0], "columns"), (
            "expecting a pandas data frame with channel / feature names as "
            "columns")
        assert self._shuffle_splits in [True, False], (
            "shuffle_splits has to be boolean")
        assert type(self._n_runs) is int and (
            self._n_runs > 0, "n_repetitions has to be an integer larger 0")
        if self._data_sets["eval"] is None and self._estimator is not None:
            assert self._n_runs >= 2, "need at least two splits for cv"
        assert type(self._n_jobs) is int and self._n_jobs >= -1, (
            "n_jobs has to be an integer larger or equal to -1")
        if self._feature_modifier is not None:
            assert callable(self._feature_modifier), (
                "modifier has to be a callable")
        if hasattr(self._estimator, "n_jobs"):
            self._estimator.n_jobs = self._n_jobs
        if self._metrics is not None and not hasattr(self._metrics, "__len__"):
            self._metrics = [self._metrics]
        if self._scaler is not None:
            scaling_functions = ["fit_transform", "transform"]
            for scaling_function in scaling_functions:
                assert hasattr(self._scaler, scaling_function), (
                    "scaler is not following scikit-learn api ({})"
                    .format(scaling_function))
        if self._estimator is not None:
            decoding_functions = ["fit", "predict"]
            for decoding_function in decoding_functions:
                assert hasattr(self._estimator, decoding_function), (
                    "classifier is not following scikit-learn api ({})"
                    .format(decoding_function))
        if self._pca_thresh:
            assert type(self._pca_thresh) in [int, float], (
                "pca_thresh has to be either int or float")
            if self._scaler is None:
                logging.warning("using pca on unscaled features")
        if self._estimator is self._preproc_f is self._feat_gen_f is None:
            logging.warning("specify 'preproc_function' to do preprocessing, "
                            "'feature_generation_function' to generate "
                            "features or 'estimator' to make predictions"
                            .format(""))
        if self._data_sets["eval"] is None:
            self._preprocessed.pop("eval")
            self._data_sets.pop("eval")
            self._features.pop("eval")
            self._targets.pop("eval")
            self.info.pop("eval")
