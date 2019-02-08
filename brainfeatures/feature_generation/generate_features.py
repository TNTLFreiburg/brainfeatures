from collections import OrderedDict
import logging

import pandas as pd
import numpy as np

from brainfeatures.utils.data_util import (
    split_into_epochs, apply_window_function, filter_to_frequency_bands,
    reject_windows_with_outliers)
from brainfeatures.feature_generation.wavelet_feature_generator import (
    WaveletFeatureGenerator)
from brainfeatures.feature_generation.frequency_feature_generator import (
    FrequencyFeatureGenerator)
from brainfeatures.feature_generation.phase_feature_generator import (
    PhaseFeatureGenerator)
from brainfeatures.feature_generation.time_feature_generator import (
    TimeFeatureGenerator)
from brainfeatures.utils.data_util import assemble_overlapping_band_limits


default_feature_generation_params = {
    "epoch_duration_s": 6,
    "max_abs_val": 800,
    "window_name": "blackmanharris",
    "band_limits": [[0, 2], [2, 4],  [4, 8], [8, 13],
                    [13, 18],  [18, 24], [24, 30], [30, 49.9]],
    "agg_mode": "median",
    "discrete_wavelet": "db4",
    "continuous_wavelet": "morl",
    "band_overlap": True
}


def run_checks(band_limits, sfreq, epoch_duration_s, agg_mode, window_name,
               domains):
    nyquist_freq = sfreq / 2
    assert np.sum(np.array(band_limits) > nyquist_freq) == 0, \
        "Cannot have a frequency band limit higher than Nyquist frequency"\
        .format(nyquist_freq)
    bin_size = 1./epoch_duration_s
    band_widths = [band_limit[1] - band_limit[0] for band_limit in band_limits]
    band_widths = np.array(band_widths)
    assert np.sum(band_widths < bin_size) == 0, \
        "Cannot have frequency bands smaller than bin size {}".format(bin_size)
    assert agg_mode in ["mean", "median", "var", "None", "none", None], \
        "Unknown aggregation mode {}".format(agg_mode)
    assert window_name in ["boxcar", "blackmanharris", "hamming", "hann",
                           "flattop", "triangle"], \
        "Cannot have a window {}".format(window_name)
    valid_domains = ["cwt", "dwt", "dft", "meta", "phase", "time", "all"]
    if type(domains) is not list:
        assert domains in valid_domains
    else:
        for domain in domains:
            assert domain in valid_domains


def generate_features_of_one_file(signals: pd.DataFrame, sfreq: int,
                                  epoch_duration_s: int, max_abs_val: int,
                                  window_name: str, band_limits: list,
                                  agg_mode: str, discrete_wavelet: str,
                                  continuous_wavelet: str, band_overlap: bool,
                                  domains="all") -> pd.DataFrame:
    # run checks / assertions
    run_checks(band_limits, sfreq, epoch_duration_s, agg_mode, window_name,
               domains)
    non_overlapping_bands = band_limits
    if band_overlap:
        band_limits = assemble_overlapping_band_limits(band_limits)
    if type(domains) is not list:
        domains = [domains]
    if agg_mode in ["none", "None", None]:
        agg_mode = None
    else:
        agg_mode = getattr(np, agg_mode)

    channels = signals.index
    signals = np.array(signals)

    # split into epochs
    epochs = split_into_epochs(
        signals=signals, sfreq=sfreq, epoch_duration_s=epoch_duration_s)
    # reject windows with outliers
    outlier_mask = reject_windows_with_outliers(
        outlier_value=max_abs_val, epochs=epochs)
    epochs = epochs[outlier_mask == False]
    # inform and return if all epochs were removed
    if epochs.size == 0:
        logging.warning("removed all epochs due to outliers")
        return None
    # weight the samples by a window function
    weighted_epochs = apply_window_function(
        epochs=epochs, window_name=window_name)

    generators = OrderedDict()
    params = OrderedDict()
    if "cwt" in domains or "all" in domains:
        name = "CWTFeatureGenerator"
        wfg = WaveletFeatureGenerator(elecs=channels, agg=agg_mode, sfreq=sfreq,
                                      domain="cwt", wavelet=continuous_wavelet,
                                      band_limits=non_overlapping_bands)
        generators.update({name: wfg})
        params.update({name: weighted_epochs})

    if "dwt" in domains or "all" in domains:
        name = "DWTFeatureGenerator"
        wfg = WaveletFeatureGenerator(agg=agg_mode, elecs=channels, sfreq=sfreq,
                                      domain="dwt", wavelet=discrete_wavelet,
                                      band_limits=non_overlapping_bands)
        generators.update({name: wfg})
        params.update({name: weighted_epochs})

    if "dft" in domains or "all" in domains:
        name = "DFTFeatureGenerator"
        ffg = FrequencyFeatureGenerator(agg=agg_mode, bands=band_limits,
                                        elecs=channels, sfreq=sfreq)
        generators.update({name: ffg})
        params.update({name: weighted_epochs})

    if "phase" in domains or "all" in domains:
        band_signals = filter_to_frequency_bands(
            signals=signals, bands=band_limits, sfreq=sfreq)
        band_epochs = split_into_epochs(band_signals, sfreq=sfreq,
                                        epoch_duration_s=epoch_duration_s)
        band_epochs = band_epochs[outlier_mask == False]

        name = "PhaseFeatureGenerator"
        pfg = PhaseFeatureGenerator(agg=agg_mode, bands=band_limits,
                                    elecs=channels)
        generators.update({name: pfg})
        params.update({name: band_epochs})

    if "time" in domains or "all" in domains:
        name = "TimeFeatureGenerator"
        tfg = TimeFeatureGenerator(agg=agg_mode, elecs=channels, sfreq=sfreq,
                                   outlier_value=max_abs_val)
        generators.update({name: tfg})
        params.update({name: epochs})

    all_features, feature_labels = [], []
    for fg_name, fg in generators.items():
        fg_params = params[fg_name]
        features = fg.generate_features(fg_params)
        labels = fg.get_feature_labels()
        all_features.append(features)
        feature_labels.extend(list(labels))

    axis = 1 if agg_mode in ["none", "None", None] else 0
    all_features = np.concatenate(all_features, axis=axis)
    if all_features.ndim == 1:
        all_features = [all_features]
    df = pd.DataFrame(all_features, columns=feature_labels)
    return df
