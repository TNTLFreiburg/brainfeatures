from scipy.signal import hilbert
import numpy as np


def instantaneous_phases(band_signals, axis):
    # use already epoched signal here?
    analytical_signal = hilbert(band_signals, axis=axis)
    return np.unwrap(np.angle(analytical_signal), axis=axis)


def phase_locking_values(inst_phases):
    (n_windows, n_bands, n_signals, n_samples) = inst_phases.shape
    plvs = []
    for electrode_id1 in range(n_signals):
        # only compute upper triangle of the synchronicity matrix and fill
        # lower triangle with identical values
        # +1 since diagonal is always 1
        for electrode_id2 in range(electrode_id1+1, n_signals):
            for band_id in range(n_bands):
                plv = phase_locking_value2(
                    theta1=inst_phases[:, band_id, electrode_id1],
                    theta2=inst_phases[:, band_id, electrode_id2]
                )
                plvs.append(plv)

    # n_window x n_bands * (n_signals*(n_signals-1))/2
    plvs = np.array(plvs).T
    return plvs


# functions below are tested to compute almost the same (at least up to e-10)
# def phase_locking_value1(theta1, theta2):
#     delta = np.subtract(theta1, theta2)
#     xs_mean = np.mean(np.cos(delta), axis=-1)
#     ys_mean = np.mean(np.sin(delta), axis=-1)
#     plv = np.sqrt(xs_mean * xs_mean + ys_mean * ys_mean)
#     return plv


def phase_locking_value2(theta1, theta2):
    # NOTE: band loop, cos, sin, manual/builtin norm won the timing challenge
    # however, this might be different for varying lengths of signals...
    delta = np.subtract(theta1, theta2)
    xs_mean = np.mean(np.cos(delta), axis=-1)
    ys_mean = np.mean(np.sin(delta), axis=-1)
    plv = np.linalg.norm([xs_mean, ys_mean], axis=0)
    return plv


# def phase_locking_value3(theta1, theta2):
#     delta = np.subtract(theta1, theta2)
#     complex_delta = np.exp(np.complex(0, 1) * delta)
#     plv = np.abs(np.mean(complex_delta, axis=-1))
#     return plv
