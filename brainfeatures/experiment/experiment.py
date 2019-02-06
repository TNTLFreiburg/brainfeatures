from datetime import datetime, date
from collections import OrderedDict
from functools import partial
import logging
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

from brainfeatures.feature_generation.generate_features import (
    generate_features_of_one_file as feat_gen_f,
    default_feature_generation_params as feat_gen_params)
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
            clf=RandomForestClassifier(n_estimators=100),
            metrics=accuracy_score,
            eval_set=None,
            n_jobs: int=1,
            preproc_function: callable=None,
            preproc_params: dict=None,
            feature_generation_function: callable=feat_gen_f,
            feature_generation_params: dict=feat_gen_params,
            n_splits_or_repetitions: int=5,
            shuffle_splits: bool=False,
            pca_thresh: float=None,
            scaler=StandardScaler(),
            feature_vector_modifier: callable=None,
            analyze_features: bool=True,
            verbosity: str="INFO"):

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
        self._verbosity = verbosity
        self._metrics = metrics
        self._n_jobs = n_jobs
        self._scaler = scaler
        self._clf = clf

        self._features = {"devel": [], "eval": []}
        self._targets = {"devel": [], "eval": []}
        self._cleaned = {"devel": [], "eval": []}
        self.info = {"devel": {}, "eval": {}}
        self._feature_names = None
        self.performances = {}
        self.predictions = {}
        self.times = {}
    """
    Class that performs one feature-based experiment on development (and
    evaluation) set.

    It is structured as follows:

    1. (optional) Read raw data from given data set(s) and apply given
                  cleaning rules.
    2. (optional) Take previously cleaned signals / read cleaned signals from
                  given data set(s) and apply given feature generation
                  procedure.
    3. (optional) Take previously generated features / read generated features
                  from given data set(s) and run (cross)-validation or final
                  evaluation using given classifier and evaluate it using
                  given metric(s).

    Parameters
    ----------
    devel_set: :class:`.DataSet`
        with __len__ and __getitem__ returning (example, sfreq, label),
    clf:  object, optional
        a classifier following scikit-learn api
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
    pca_thresh: integer, float, optional
        inter specifying number of components to keep / float specifying
        percentage of explained variance to determine number of components to
        keep with application of principal component analysis
    scaler: object, optional
        a scaler following scikit-learn api used to scale feature values
    verbosity: str, optional
        verbosity level
    """

    def _run_checks(self):
        """
        Assure conformity of given arguments.
        """
        assert self._verbosity in ["DEBUG", "INFO", "WARNING", "ERROR", 0,
                                   10, 20, 30, 40], "unknown verbosity level"
        if self._feat_gen_f is not None:
            assert hasattr(self._feat_gen_f, "__call__"), (
                "feature_generation_procedure has to be a callable")
        if self._preproc_f is not None:
            assert hasattr(self._preproc_f, "__call__"), (
                "cleaning_procedure has to be a callable")
        if self._preproc_params is not None:
            assert type(self._preproc_params) is dict, (
                "cleaning_params has to be a dictionary")
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
            self._n_runs > 0, "n_repetitions has to be an integer larger than "
                              "0")
        if self._data_sets["eval"] is None:
            assert self._n_runs >= 2, "need at least two splits for cv"
        assert type(self._n_jobs) is int and self._n_jobs >= -1, (
            "n_jobs has to be an integer larger or equal to -1")
        if self._feature_modifier is not None:
            assert callable(self._feature_modifier), (
                "modifier has to be a callable")
        if hasattr(self._clf, "n_jobs"):
            self._clf.n_jobs = self._n_jobs
        if self._metrics is not None and not hasattr(self._metrics, "__len__"):
            self._metrics = [self._metrics]
        if self._scaler is not None:
            scaling_functions = ["fit_transform", "transform"]
            for scaling_function in scaling_functions:
                assert hasattr(self._scaler, scaling_function), (
                    "scaler is not following scikit-learn api ({})"
                    .format(scaling_function))
        if self._clf is not None:
            decoding_functions = ["fit", "predict"]
            for decoding_function in decoding_functions:
                assert hasattr(self._clf, decoding_function), (
                    "classifier is not following scikit-learn api ({})"
                    .format(decoding_function))
        if self._pca_thresh:
            assert type(self._pca_thresh) in [int, float], (
                "pca_thresh has to be either int or float")
            if self._scaler is None:
                logging.warning("using pca on unscaled features")
        if "eval" in self._data_sets and self._data_sets["eval"] is None:
            self._data_sets.pop("eval")
            self._features.pop("eval")
            self._targets.pop("eval")
            self._cleaned.pop("eval")
            self.info.pop("eval")

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
        logging.info("Making clean ({})".format(set_name))
        if self._preproc_params is not None:
            self._preproc_f = partial(self._preproc_f, **self._preproc_params)

        x_fs_y_pre = Parallel(n_jobs=self._n_jobs)(
            (delayed(self._preproc_f)(x, fs, y)
             for (x, fs, y) in self._data_sets[set_name]))

        for (x_pre, fs_pre, y_pre) in x_fs_y_pre:
            self._cleaned[set_name].append(x_pre)
            self._targets[set_name].append(y_pre)
        if "sfreq" not in self.info[set_name]:
            self.info[set_name]["sfreq"] = fs_pre
        self.times.setdefault("cleaning", {}).update(
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
            either "clean" or "features"
        """
        start = time.time()
        logging.info("Loading {} ({})".format(set_name, set_state))
        for (x, fs, y) in self._data_sets[set_name]:
            getattr(self, "_" + set_state)[set_name].append(x)
            self._targets[set_name].append(y)
        # store feature names and sampling frequency
        if "sfreq" not in self.info[set_name]:
            self.info[set_name]["sfreq"] = fs
        if self._feature_names is None:
            self._feature_names = list(x.columns)
        self.times.setdefault("loading", {}).update(
            {set_name: time.time() - start})

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
        if self._feat_gen_params is not None:
            self._feat_gen_f = partial(self._feat_gen_f,
                                       **self._feat_gen_params)
        feature_matrix = Parallel(n_jobs=self._n_jobs)(
            (delayed(self._feat_gen_f)(example, self.info[set_name]["sfreq"])
             for example in self._cleaned[set_name]))
        for i, feature_vector in enumerate(feature_matrix):
            if feature_vector is not None:
                self._features[set_name].append(feature_vector)
            # important: if feature generation fails, and therefore feature
            # vector is None, remove according label!
            else:
                del self._targets[set_name][i]
                logging.warning("removed example {} from labels".format(i))
        if self._feature_names is None:
            self._feature_names = list(feature_vector.columns)
        assert len(self._features[set_name]) == len(self._targets[set_name]), (
            "number of feature vectors does not match number of labels")
        self.times.setdefault("feature generation", {}).update(
            {set_name: time.time() - start})

    def _validate(self):
        """
        Perform (cross-)validation on development set.
        """
        start = time.time()
        logging.info("Making predictions (validation)")
        assert len(self._features["devel"]) == len(self._targets["devel"]), (
            "number of devel examples and labels differs!")
        validation_results, info = validate(
            self._features["devel"], self._targets["devel"], self._clf,
            self._n_runs, self._shuffle_splits, self._scaler,
            self._pca_thresh, self._feature_anaylsis)
        self.predictions.update(validation_results)
        self.info.update(info)
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
        set_names = []
        if set_name == "devel":
            set_names.extend(["train"])
        set_names.extend([set_name])

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
            "number of eval examples and labels differs!")
        eval_results, eval_info = final_evaluate(
            self._features["devel"], self._targets["devel"],
            self._features["eval"], self._targets["eval"], self._clf,
            self._n_runs, self._scaler, self._pca_thresh,
            self._feature_anaylsis)
        self.predictions.update(eval_results)
        self.info.update(eval_info)
        self.times["final evaluation"] = time.time() - start

    def _decode(self, set_name):
        """
        Run decoding on data specified by set_name, analyze the performance
        and the features used.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        """
        # TODELAY: impove feature vector modifier
        if self._feature_modifier is not None:
            self._features[set_name], self._feature_names =\
                self._feature_modifier(self._data_sets[set_name],
                                       self._features[set_name],
                                       self._feature_names)
            assert len(self._features[set_name]) > 0, (
                "removed all feature vectors")
            assert self._features[set_name][0].shape[-1] == (
                len(self._feature_names), "number of features and feature "
                                          "names does not match")
        if set_name == "devel":
            self._validate()
        else:
            assert set_name == "eval", "unknown set name"
            self._final_evaluate()

        if self._metrics is not None:
            self._analyze_performance(set_name)

    def _analyze_features(self, set_name):
        """
        Perform analysis of features specified by set_name using correlation
        coefficient, principle components and feature importances.

        Parameters
        ----------
        set_name: str
            either "devel" or "eval"
        """
        # always analyze correlation of features
        feature_correlations = analyze_feature_correlations(
            self._features[set_name])
        self.info[set_name].update({"feature_correlations":
                                    feature_correlations})
        if set_name == "devel":
            set_name = "valid"
        # if using pca, analyze principal components
        if self._pca_thresh is not None:
            pca_features = analyze_pca_components(
                self.info[set_name]["pca_components"])
            self.info[set_name].update({"pca_features": pca_features})

        # do feature importances with principle components?
        if self._pca_thresh is None:
            # if using random forest and not pca, analyze feature_importances
            if "feature_importances" in self.info[set_name]:
                analyze_feature_importances(
                    self.info[set_name]["feature_importances"])

            # if using random forest and not pca, analyze feature_importances
            if "rfpimp_importances" in self.info[set_name]:
                analyze_feature_importances(
                    self.info[set_name]["rfpimp_importances"])

    def analyze_features(self, set_names):
        """
        Analyze features of given sets.

        Parameters
        ----------
        set_names: list
            list of strings specifying the set name to be analyzed
        """
        for set_name in set_names:
            assert set_name in self.info.keys(), (
                "unkown set name {}".format(set_name))
            self._analyze_features(set_name)

    def run(self):
        """
        Run complete experiment.
        """
        today, now = date.today(), datetime.time(datetime.now())
        logging.info('Started on {} at {}'.format(today, now))

        self._run_checks()
        do_pre = self._preproc_f is not None
        do_features = self._feat_gen_f is not None
        do_predictions = self._clf is not None and self._features["devel"]
        for set_name in self._data_sets.keys():
            if do_pre:
                self._preprocess(set_name)

            if do_features:
                if not do_pre:
                    self._load(set_name, "clean")
                self._generate_features(set_name)

            if not do_pre and not do_features:
                self._load(set_name, "features")

            if do_predictions:
                self._decode(set_name)

        today, now = date.today(), datetime.time(datetime.now())
        logging.info("Finished on {} at {}.".format(today, now))
