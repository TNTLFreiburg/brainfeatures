from brainfeatures.feature_generation.abstract_feature_generator import (
    AbstractFeatureGenerator)
from brainfeatures.feature_generation import features_phase


class PhaseFeatureGenerator(AbstractFeatureGenerator):
    """ computes features in the time domain implemented in features_time """

    def get_feature_labels(self):
        """
        :return: list of feature labels of the form <domain>_<feature>_<channel>
        """
        feature_labels = []
        for sync_feat in self.sync_feats:
            for band_id, band in enumerate(self.bands):
                lower, upper = band
                for electrode_id, electrode in enumerate(self.electrodes):
                    for electrode_id2 in range(electrode_id + 1,
                                               len(self.electrodes)):
                        label = '_'.join([
                            self.domain,
                            sync_feat,
                            '-'.join([str(lower), str(upper) + 'Hz',
                                      str(electrode),
                                      str(self.electrodes[electrode_id2])])
                        ])
                        feature_labels.append(label)
        return feature_labels

    def generate_features(self, band_epochs):
        # n_windows x n_bands x n_elecs x n_samples_in_window
        epochs_instantaneous_phases = features_phase.instantaneous_phases(
            band_signals=band_epochs, axis=-1)

        # n_windows x n_bands x n_signals*(n_signals-1)/2
        phase_locking_values = features_phase.phase_locking_values(
            inst_phases=epochs_instantaneous_phases)

        # aggregate over dimension of epochs
        if self.agg_mode:
            # n_bands x n_signals*(n_signals-1)/2
            phase_locking_values = self.agg_mode(phase_locking_values, axis=0)

        return phase_locking_values

    def __init__(self, elecs, agg, bands, domain="phase"):
        super(PhaseFeatureGenerator, self).__init__(
            domain=domain, electrodes=elecs, agg_mode=agg)
        self.sync_feats = ["plv"]
        self.bands = bands
