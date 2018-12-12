from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from datetime import datetime, date
from collections import OrderedDict
from functools import partial
import logging
import time

from brainfeaturedecode.feature_generation.generate_features import \
    generate_features_of_one_file, default_feature_generation_params
from brainfeaturedecode.analysis.analyze import analyze_quality_of_predictions, \
    analyze_feature_importances, analyze_feature_correlations, \
    analyze_pca_components
from brainfeaturedecode.decoding.decode import validate, final_evaluate

# TODO: add cropping feature vector?
# TODO: move agg mode out of feature generators to experiment? -> moves a lot of data around
# TODO: make sure to not run when feature generation procedure is specified but data set is raw? how to know? -> no, it's fine to not apply cleaning rules
# TODO: free memory? how much memory is needed?
# TODO: use feature labels / load feature labels? -> no this will be done in analysis
# TODO: split train into train/test before cleaning/feature generation?
# TODO: make meta feature generation usable with experiment class?
# TODO: add simple tuning to find hyperparameters?


class Experiment(object):
    """
    Class that performs one feature-based experiment on training (and
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
    train_set: :class:`.DataSet`
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
    cleaning_procedure: function, optional
        takes signals and sampling frequency of the signals and returns cleaned
        signals
    cleaning_params: dict, optional
        keyword arguments needed for cleaning functions except signals and
        sampling frequency in the form
    feature_generation_procedure: function, optional
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
    def __init__(self, train_set, clf=RandomForestClassifier(n_estimators=100),
                 metrics=accuracy_score, eval_set=None, n_jobs=1,
                 cleaning_procedure=None, cleaning_params=None,
                 feature_generation_procedure=generate_features_of_one_file,
                 feature_generation_params=default_feature_generation_params,
                 n_splits_or_repetitions=5, shuffle_splits=False,
                 pca_thresh=None, scaler=StandardScaler(),
                 feature_vector_modifier=None, verbosity="INFO"):
        self.data_sets = OrderedDict([("train", train_set), ("eval", eval_set)])
        self.feature_generation_procedure = feature_generation_procedure
        self.feature_generation_params = feature_generation_params
        self.n_splits_or_repetitions = n_splits_or_repetitions
        self.cleaning_procedure = cleaning_procedure
        self.cleaning_params = cleaning_params
        self.shuffle_splits = shuffle_splits
        self.pca_thresh = pca_thresh
        self.verbosity = verbosity
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.scaler = scaler
        self.clf = clf

        self.features = {"train": [], "eval": []}
        self.targets = {"train": [], "eval": []}
        self.clean = {"train": [], "eval": []}
        self.info = {"train": {}, "eval": {}}
        self.feature_vector_modifier = feature_vector_modifier
        self.feature_labels = None
        self.performances = {}
        self.predictions = {}
        self.times = {}

    def _run_checks(self):
        """
        Assure conformity of given arguments.
        """
        assert not (self.cleaning_procedure is None
                    and self.feature_generation_procedure is None
                    and self.clf is None), "please specify what to do"
        assert self.verbosity in ["DEBUG", "INFO", "WARNING", "ERROR",
                                  0, 10, 20, 30, 40], "unknown verbosity level"
        if self.feature_generation_procedure is not None:
            assert hasattr(self.feature_generation_procedure, "__call__"), \
                "feature_generation_procedure has to be a callable"
        if self.cleaning_procedure is not None:
            assert hasattr(self.cleaning_procedure, "__call__"), \
                "cleaning_procedure has to be a callable"
        if self.cleaning_params is not None:
            assert type(self.cleaning_params) is dict, \
                "cleaning_params has to be a dictionary"
        if self.feature_generation_params is not None:
            assert type(self.feature_generation_params) is dict, \
                "feature_generation_params has to be a dictionary"
        assert len(self.data_sets["train"][0]) == 3, \
            "__getitem__ of data set needs to return x, fs, y"
        assert self.shuffle_splits in [True, False], \
            "shuffle_splits has to be boolean"
        assert type(self.n_splits_or_repetitions) is int and \
               self.n_splits_or_repetitions > 0, \
            "n_repetitions has to be an integer larger than 0"
        if self.data_sets["eval"] is None:
            assert self.n_splits_or_repetitions >= 2, \
                "need at least two splits for cv"
        assert type(self.n_jobs) is int and self.n_jobs >= -1, \
            "n_jobs has to be an integer larger or equal to -1"
        if self.feature_vector_modifier is not None:
            assert callable(self.feature_vector_modifier), \
                "modifier has to be a callable"
        if hasattr(self.clf, "n_jobs"):
            self.clf.n_jobs = self.n_jobs
        if self.metrics is not None and not hasattr(self.metrics, "__len__"):
            self.metrics = [self.metrics]
        if self.scaler is not None:
            scaling_functions = ["fit_transform", "transform"]
            for scaling_function in scaling_functions:
                assert hasattr(self.scaler, scaling_function), \
                    "scaler is not following scikit-learn api ({})" \
                        .format(scaling_function)
        if self.clf is not None:
            decoding_functions = ["fit", "predict"]
            for decoding_function in decoding_functions:
                assert hasattr(self.clf, decoding_function), \
                    "classifier is not following scikit-learn api ({})"\
                        .format(decoding_function)
        if self.pca_thresh:
            assert type(self.pca_thresh) in [int, float], \
                "pca_thresh has to be either int or float"
            if self.scaler is None:
                logging.warning("using pca on unscaled features")
        if "eval" in self.data_sets and self.data_sets["eval"] is None:
            self.data_sets.pop("eval")
            self.features.pop("eval")
            self.targets.pop("eval")
            self.clean.pop("eval")
            self.info.pop("eval")

    def _clean(self, train_or_eval):
        """
        Apply given cleaning rules to all examples in data set specified by
        train_or_eval.

        Parameters
        ----------
        train_or_eval: str
            either "train" or "eval"
        """
        start = time.time()
        logging.info("Making clean ({})".format(train_or_eval))
        if self.cleaning_params is not None:
            self.cleaning_procedure = partial(self.cleaning_procedure,
                                              **self.cleaning_params)
        cleaned_signals_and_sfreq = Parallel(n_jobs=self.n_jobs)\
            (delayed(self.cleaning_procedure)(example, sfreq)
             for (example, sfreq, label) in self.data_sets[train_or_eval])
        # not nice, iterating twice
        for (cleaned_signals, sfreq) in cleaned_signals_and_sfreq:
            if "sfreq" not in self.info[train_or_eval]:
                self.info[train_or_eval]["sfreq"] = sfreq
            self.clean[train_or_eval].append(cleaned_signals)
        self.targets[train_or_eval] = self.data_sets[train_or_eval].targets
        self.times.setdefault("cleaning", {}).update(
            {train_or_eval: time.time() - start})

    def _load_cleaned_or_features(self, train_or_eval, clean_or_features):
        """
        Load cleaned signals or features from data set specified by
        train_or_eval.

        Parameters
        ----------
        train_or_eval: str
            either "train" or "eval"
        clean_or_features: str
            either "clean" or "features"
        """
        start = time.time()
        logging.info("Loading {} ({})".format(train_or_eval,
                                              clean_or_features))
        for (data, sfreq, label) in self.data_sets[train_or_eval]:
            getattr(self, clean_or_features)[train_or_eval].append(data)
            self.targets[train_or_eval].append(label)
            if "sfreq" not in self.info[train_or_eval]:
                self.info[train_or_eval]["sfreq"] = sfreq
        if hasattr(self.data_sets[train_or_eval], "feature_labels") \
                and getattr(self.data_sets[train_or_eval], "feature_labels") is not None:
            self.feature_labels = self.data_sets[train_or_eval].feature_labels
        self.times.setdefault("loading", {}).update(
            {train_or_eval: time.time() - start})

    def _generate_features(self, train_or_eval):
        """
        Apply given feature generation procedure to all examples in data set
        specified by train_or_eval.

        Parameters
        ----------
        train_or_eval: str
            either "train" or "eval"
        """
        start = time.time()
        logging.info("Generating features ({})".format(train_or_eval))
        if self.feature_generation_params is not None:
            self.feature_generation_procedure = partial(
                self.feature_generation_procedure,
                **self.feature_generation_params)

        feature_vectors = Parallel(n_jobs=self.n_jobs)\
            (delayed(self.feature_generation_procedure)
             (example, self.info[train_or_eval]["sfreq"])
             for example in self.clean[train_or_eval])

        for i, feature_vector in enumerate(feature_vectors):
            if feature_vector is not None:
                if self.feature_labels is None:
                    self.feature_labels = list(feature_vector.columns)
                self.features[train_or_eval].append(feature_vector.values)
            # important: if feature generation fails, and therefore feature
            # vector is None remove according label!
            else:
                del self.targets[train_or_eval][i]
                logging.warning("removed example {} from labels".format(i))
        assert len(self.features[train_or_eval]) == \
               len(self.targets[train_or_eval]), \
            "number of feature vectors does not match number of labels"
        self.times.setdefault("feature generation", {}).update(
            {train_or_eval: time.time() - start})

    def _validate(self):
        """
        Perform (cross-)validation on training set.
        """
        start = time.time()
        logging.info("Making predictions (validation)")
        assert len(self.features["train"]) == len(self.targets["train"]), \
            "number of train examples and labels differs!"
        validation_results, info = validate(
            self.features["train"], self.targets["train"], self.clf,
            self.n_splits_or_repetitions, self.shuffle_splits, self.scaler,
            self.pca_thresh)
        self.predictions.update({"train": validation_results["predictions"]})
        self.info["train"] = info
        self.times["validation"] = time.time() - start

    def _analyze_performance(self, train_or_eval):
        """
        Apply specified metrics on predictions on data set specified by
        train_or_eval.

        Parameters
        ----------
        train_or_eval: str
            either "train" or "eval"
        """
        if train_or_eval == "train":
            valid_or_final_evaluation = "validation"
        else:
            valid_or_final_evaluation = "final evaluation"
        logging.info("Computing performances ({})".format(
            valid_or_final_evaluation))
        performances = analyze_quality_of_predictions(
            self.predictions[train_or_eval], self.metrics)
        self.performances.update({train_or_eval: performances})
        logging.info("Achieved in average\n{}\n".format(
            self.performances[train_or_eval].mean().to_string()))

    def _final_evaluate(self):
        """
        Perform final evaluation on training and final evaluation set.
        """
        start = time.time()
        logging.info("Making predictions (final evaluation)")
        assert len(self.features["eval"]) == len(self.targets["eval"]), \
            "number of eval examples and labels differs!"
        evaluation_results, eval_info = final_evaluate(
            self.features["train"], self.targets["train"], self.features["eval"],
            self.targets["eval"], self.clf, self.n_splits_or_repetitions,
            self.scaler, self.pca_thresh)
        self.predictions.update({"eval": evaluation_results["predictions"]})
        self.info["eval"] = eval_info
        self.times["final evaluation"] = time.time() - start

    def run(self):
        """
        Run complete experiment.
        """
        log = logging.getLogger()
        log.setLevel("INFO")
        today, now = date.today(), datetime.time(datetime.now())
        logging.info('Started on {} at {}'.format(today, now))

        self._run_checks()
        do_clean = self.cleaning_procedure is not None
        do_features = self.feature_generation_procedure is not None
        do_predictions = self.clf is not None
        for train_or_eval in self.data_sets.keys():
            if do_clean:
                self._clean(train_or_eval)

            if do_features:
                if not do_clean:
                    self._load_cleaned_or_features(train_or_eval, "clean")
                self._generate_features(train_or_eval)

            if not do_clean and not do_features:
                self._load_cleaned_or_features(train_or_eval, "features")

            # if "eval" is set, don't run cv?
            # and "eval" not in self.data_sets \
            if train_or_eval == "train" \
                    and do_predictions:
                # TODO: don't give self there as an argument
                if self.feature_vector_modifier is not None:
                    self.feature_vector_modifier(self)
                self._validate()
                if self.metrics is not None:
                    self._analyze_performance(train_or_eval)

            elif train_or_eval == "eval" and do_predictions:
                # TODO: don't give self there as an argument
                if self.feature_vector_modifier is not None:
                    self.feature_vector_modifier(self)
                self._final_evaluate()
                if self.metrics is not None:
                    self._analyze_performance(train_or_eval)

        today, now = date.today(), datetime.time(datetime.now())
        logging.info("Finished on {} at {}.".format(today, now))
