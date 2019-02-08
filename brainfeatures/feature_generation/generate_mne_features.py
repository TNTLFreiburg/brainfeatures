import logging

from mne_features.feature_extraction import extract_features
import numpy as np

from brainfeatures.utils.data_util import (split_into_epochs,
                                           reject_windows_with_outliers,
                                           assemble_overlapping_band_limits)

BAND_LIMITS = np.array(
        [[0, 2], [2, 4],  [4, 8], [8, 13],
         [13, 18],  [18, 24], [24, 30], [30, 49.9]])
default_mne_feature_generation_params = {
    # univariate
    'pow_freq_bands__freq_bands': assemble_overlapping_band_limits(
        BAND_LIMITS),
    'pow_freq_bands__normalize': True,
    'pow_freq_bands__ratios': None,
    'app_entropy__emb': 2,
    'app_entropy__metric': 'chebyshev',
    'samp_entropy__emb': 2,
    'samp_entropy__metric': 'chebyshev',
    'hjorth_mobility_spect__normalize': False,
    'hjorth_complexity_spect__normalize': False,
    'higuchi_fd__kmax': 5,
    'spect_slope__fmin': 0.1,
    'spect_slope__fmax': 50,
    'spect_slope__with_intercept': True,
    'svd_entropy__tau': 2,
    'svd_entropy__emb': 10,
    'svd_fisher_info__tau': 2,
    'svd_fisher_info__emb': 10,
    'energy_freq_bands__freq_bands': assemble_overlapping_band_limits(
        BAND_LIMITS),
    'energy_freq_bands__deriv_filt': True,
    'spect_edge_freq__ref_freq': None,
    'spect_edge_freq__edge': None,
    'wavelet_coef_energy__wavelet_name': 'db4',
    'teager_kaiser_energy__wavelet_name': 'db4',
    # bivariate
    'max_cross_corr__include_diag': False,
    'phase_lock_val__include_diag': False,
    'nonlin_interdep__tau': 2,
    'nonlin_interdep__emb': 10,
    'nonlin_interdep__nn': 5,
    'nonlin_interdep__include_diag': False,
    'time_corr__with_eigenvalues': True,
    'time_corr__include_diag': False,
    'spect_corr__db': False,
    'spect_corr__with_eigenvalues': True,
    'spect_corr__include_diag': False,
}


def generate_mne_features_of_one_file(signals, sfreq, selected_funcs,
                                      func_params, epoch_duration_s,
                                      max_abs_val, agg_mode):
    if agg_mode in ["none", "None", None]:
        agg_mode = None
    else:
        getattr(np, agg_mode)

    epochs = split_into_epochs(signals=signals, sfreq=sfreq,
                               epoch_duration_s=epoch_duration_s)
    mask = reject_windows_with_outliers(epochs, outlier_value=max_abs_val)
    epochs = epochs[mask == False]
    if epochs.size == 0:
        logging.warning("removed all epochs due to outliers")
        return None, None

    # generate features implemented in mne_features
    features = extract_features(
        epochs, sfreq, selected_funcs, funcs_params=func_params,
        return_as_df=True)
    # aggregate over dimension of epochs
    if agg_mode:
        features = agg_mode(features, axis=0)

    return features
