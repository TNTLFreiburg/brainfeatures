# Inspired by implementation of Manuel Blum and pyeeg library
from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew
import numpy as _np


def _embed_seq(X, Tau, D):
    # taken from pyeeg
    """Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    Parameters
    ----------

    X
        list

        a time series

    Tau
        integer

        the lag or delay when building embedding sequence

    D
        integer

        the embedding dimension

    Returns
    -------

    Y
        2-D list

        embedding matrix built

    Examples
    ---------------
    >>> import pyeeg
    >>> a=range(0,9)
    >>> pyeeg.embed_seq(a,1,4)
    array([[ 0.,  1.,  2.,  3.],
           [ 1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.],
           [ 3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.],
           [ 5.,  6.,  7.,  8.]])
    >>> pyeeg.embed_seq(a,2,3)
    array([[ 0.,  2.,  4.],
           [ 1.,  3.,  5.],
           [ 2.,  4.,  6.],
           [ 3.,  5.,  7.],
           [ 4.,  6.,  8.]])
    >>> pyeeg.embed_seq(a,4,1)
    array([[ 0.],
           [ 1.],
           [ 2.],
           [ 3.],
           [ 4.],
           [ 5.],
           [ 6.],
           [ 7.],
           [ 8.]])

    """
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return _np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def detrended_fluctuation_analysis(epochs, axis, **kwargs):
    def dfa_1d(X, Ave=None, L=None):
        # Taken from pyeeg
        """Compute Detrended Fluctuation Analysis from a time series X and length of
        boxes L.

        The first step to compute DFA is to integrate the signal. Let original
        series be X= [x(1), x(2), ..., x(N)].

        The integrated signal Y = [y(1), y(2), ..., y(N)] is obtained as follows
        y(k) = \sum_{i=1}^{k}{x(i)-Ave} where Ave is the mean of X.

        The second step is to partition/slice/segment the integrated sequence Y
        into boxes. At least two boxes are needed for computing DFA. Box sizes are
        specified by the L argument of this function. By default, it is from 1/5 of
        signal length to one (x-5)-th of the signal length, where x is the nearest
        power of 2 from the length of the signal, i.e., 1/16, 1/32, 1/64, 1/128,
        ...

        In each box, a linear least square fitting is employed on data in the box.
        Denote the series on fitted line as Yn. Its k-th elements, yn(k),
        corresponds to y(k).

        For fitting in each box, there is a residue, the sum of squares of all
        offsets, difference between actual points and points on fitted line.

        F(n) denotes the square root of average total residue in all boxes when box
        length is n, thus
        Total_Residue = \sum_{k=1}^{N}{(y(k)-yn(k))}
        F(n) = \sqrt(Total_Residue/N)

        The computing to F(n) is carried out for every box length n. Therefore, a
        relationship between n and F(n) can be obtained. In general, F(n) increases
        when n increases.

        Finally, the relationship between F(n) and n is analyzed. A least square
        fitting is performed between log(F(n)) and log(n). The slope of the fitting
        line is the DFA value, denoted as Alpha. To white noise, Alpha should be
        0.5. Higher level of signal complexity is related to higher Alpha.

        Parameters
        ----------

        X:
            1-D Python list or _np array
            a time series

        Ave:
            integer, optional
            The average value of the time series

        L:
            1-D Python list of integers
            A list of box size, integers in ascending order

        Returns
        -------

        Alpha:
            integer
            the result of DFA analysis, thus the slope of fitting line of log(F(n))
            vs. log(n). where n is the

        Examples
        --------
        >>> import pyeeg
        >>> from numpy.random import randn
        >>> print(pyeeg.dfa(randn(4096)))
        0.490035110345

        Reference
        ---------
        Peng C-K, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling
        exponents and crossover phenomena in nonstationary heartbeat time series.
        _Chaos_ 1995;5:82-87

        Notes
        -----

        This value depends on the box sizes very much. When the input is a white
        noise, this value should be 0.5. But, some choices on box sizes can lead to
        the value lower or higher than 0.5, e.g. 0.38 or 0.58.

        Based on many test, I set the box sizes from 1/5 of    signal length to one
        (x-5)-th of the signal length, where x is the nearest power of 2 from the
        length of the signal, i.e., 1/16, 1/32, 1/64, 1/128, ...

        You may generate a list of box sizes and pass in such a list as a
        parameter.

        """
        X = _np.array(X)

        if Ave is None:
            Ave = _np.mean(X)

        Y = _np.cumsum(X)
        Y -= Ave

        if L is None:
            L = _np.floor(len(X) * 1 / (
                    2 ** _np.array(list(range(1, int(_np.log2(len(X))) - 4))))
                            )

        F = _np.zeros(len(L))  # F(n) of different given box length n

        for i in range(0, len(L)):
            n = int(L[i])  # for each box length L[i]
            if n == 0:
                print("time series is too short while the box length is too big")
                print("abort")
                exit()
            for j in range(0, len(X), n):  # for each box
                if j + n < len(X):
                    c = list(range(j, j + n))
                    # coordinates of time in the box
                    c = _np.vstack([c, _np.ones(n)]).T
                    # the value of data in the box
                    y = Y[j:j + n]
                    # add residue in this box
                    F[i] += _np.linalg.lstsq(c, y, rcond=None)[1]
            F[i] /= ((len(X) / n) * n)
        F = _np.sqrt(F)

        stacked = _np.vstack([_np.log(L), _np.ones(len(L))])
        stacked_t = stacked.T
        Alpha = _np.linalg.lstsq(stacked_t, _np.log(F), rcond=None)

        return Alpha[0][0]

    return _np.apply_along_axis(dfa_1d, axis, epochs)


def energy(epochs, axis, **kwargs):
    return _np.mean(epochs*epochs, axis=axis)


def fisher_information(epochs, axis, **kwargs):
    def fisher_info_1d(a, tau, de):
        # taken from pyeeg improvements
        r"""
        Compute the Fisher information of a signal with embedding dimension "de" and delay "tau" [PYEEG]_.
        Vectorised (i.e. faster) version of the eponymous PyEEG function.
        :param a: a one dimensional floating-point array representing a time series.
        :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
        :param tau: the delay
        :type tau: int
        :param de: the embedding dimension
        :type de: int
        :return: the Fisher information, a scalar
        :rtype: float
        """

        mat = _embed_seq(a, tau, de)
        W = _np.linalg.svd(mat, compute_uv=False)
        W /= sum(W)  # normalize singular values
        FI_v = (W[1:] - W[:-1]) ** 2 / W[:-1]
        return _np.sum(FI_v)

    tau = kwargs["Tau"]
    de = kwargs["DE"]
    return _np.apply_along_axis(fisher_info_1d, axis, epochs, tau, de)


def fractal_dimension(epochs, axis, **kwargs):
    diff1 = _np.diff(epochs)
    sum_of_distances = _np.sum(_np.sqrt(diff1 * diff1), axis=axis)
    max_dist = _np.apply_along_axis(lambda epoch: _np.max(_np.sqrt(_np.square(epoch - epoch[0]))), axis, epochs)
    return _np.divide(_np.log10(sum_of_distances), _np.log10(max_dist))


def higuchi_fractal_dimension(epochs, axis, **kwargs):
    def hfd_1d(X, Kmax):
        # taken from pyeeg
        """ Compute Hjorth Fractal Dimension of a time series X, kmax
         is an HFD parameter
        """
        L = []
        x = []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, int(_np.floor((N - m) / k))):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / _np.floor((N - m) / float(k)) / k
                Lk.append(Lmk)
            L.append(_np.log(_np.mean(Lk)))
            x.append([_np.log(float(1) / k), 1])

        (p, r1, r2, s) = _np.linalg.lstsq(x, L, rcond=None)
        return p[0]
    Kmax = kwargs["Kmax"]
    return _np.apply_along_axis(hfd_1d, axis, epochs, Kmax)


def hjorth_activity(epochs, axis, **kwargs):
    return _np.var(epochs, axis=axis)


def hjorth_complexity(epochs, axis, **kwargs):
    diff1 = _np.diff(epochs, axis=axis)
    diff2 = _np.diff(diff1, axis=axis)
    sigma1 = _np.std(diff1, axis=axis)
    sigma2 = _np.std(diff2, axis=axis)
    return _np.divide(_np.divide(sigma2, sigma1), hjorth_mobility(epochs, axis))


def hjorth_mobility(epochs, axis, **kwargs):
    diff = _np.diff(epochs, axis=axis)
    sigma0 = _np.std(epochs, axis=axis)
    sigma1 = _np.std(diff, axis=axis)
    return _np.divide(sigma1, sigma0)


def _hjorth_parameters(epochs, axis, **kwargs):
    activity = _np.var(epochs, axis=axis)
    diff1 = _np.diff(epochs, axis=axis)
    diff2 = _np.diff(diff1, axis=axis)
    sigma0 = _np.std(epochs, axis=axis)
    sigma1 = _np.std(diff1, axis=axis)
    sigma2 = _np.std(diff2, axis=axis)
    mobility = _np.divide(sigma1, sigma0)
    complexity = _np.divide(_np.divide(sigma2, sigma1), hjorth_mobility(epochs, axis))
    return activity, complexity, mobility


def hurst_exponent(epochs, axis, **kwargs):
    def hurst_1d(X):
        # taken from pyeeg
        """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
        of the time-series is similar to random walk. If H<0.5, the time-series
        cover less "distance" than a random walk, vice verse.

        Parameters
        ----------

        X

            list

            a time series

        Returns
        -------
        H

            float

            Hurst exponent

        Notes
        --------
        Author of this function is Xin Liu

        Examples
        --------

        >>> import pyeeg
        >>> from numpy.random import randn
        >>> a = randn(4096)
        >>> pyeeg.hurst(a)
        0.5057444

        """
        X = _np.array(X)
        N = X.size
        T = _np.arange(1, N + 1)
        Y = _np.cumsum(X)
        Ave_T = Y / T

        S_T = _np.zeros(N)
        R_T = _np.zeros(N)
        for i in range(N):
            S_T[i] = _np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = _np.ptp(X_T[:i + 1])

        # check for indifferent measurements at time series start
        # they could be introduced by resampling and have to be removed
        # if not removed, it will cause division by std = 0
        for i in range(1, len(S_T)):
            if _np.diff(S_T)[i - 1] != 0:
                break
        for j in range(1, len(R_T)):
            if _np.diff(R_T)[j - 1] != 0:
                break
        k = max(i, j)
        assert k < 10, "rethink it!"

        R_S = R_T[k:] / S_T[k:]
        R_S = _np.log(R_S)

        n = _np.log(T)[k:]
        A = _np.column_stack((n, _np.ones(n.size)))
        [m, c] = _np.linalg.lstsq(A, R_S, rcond=None)[0]
        H = m
        return H
    return _np.apply_along_axis(hurst_1d, axis, epochs)


def kurtosis(epochs, axis, **kwargs):
    return _kurt(epochs, axis=axis, bias=False)


def line_length(epochs, axis, **kwargs):
    return _np.sum(_np.abs(_np.diff(epochs)), axis=axis)


def largest_lyauponov_exponent(epochs, axis, **kwargs):
    def LLE_1d(x, tau, n, T, fs):
        # taken from pyeeg
        """Calculate largest Lyauponov exponent of a given time series x using
        Rosenstein algorithm.

        Parameters
        ----------

        x
            list

            a time series

        n
            integer

            embedding dimension

        tau
            integer

            Embedding lag

        fs
            integer

            Sampling frequency

        T
            integer

            Mean period

        Returns
        ----------

        Lexp
           float

           Largest Lyapunov Exponent

        Notes
        ----------
        A n-dimensional trajectory is first reconstructed from the observed data by
        use of embedding delay of tau, using pyeeg function, embed_seq(x, tau, n).
        Algorithm then searches for nearest neighbour of each point on the
        reconstructed trajectory; temporal separation of nearest neighbours must be
        greater than mean period of the time series: the mean period can be
        estimated as the reciprocal of the mean frequency in power spectrum

        Each pair of nearest neighbours is assumed to diverge exponentially at a
        rate given by largest Lyapunov exponent. Now having a collection of
        neighbours, a least square fit to the average exponential divergence is
        calculated. The slope of this line gives an accurate estimate of the
        largest Lyapunov exponent.

        References
        ----------
        Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. "A
        practical method for calculating largest Lyapunov exponents from small data
        sets." Physica D: Nonlinear Phenomena 65.1 (1993): 117-134.


        Examples
        ----------
        >>> import pyeeg
        >>> X = _np.array([3,4,1,2,4,51,4,32,24,12,3,45])
        >>> pyeeg.LLE(X,2,4,1,1)
        >>> 0.18771136179353307

        """

        Em = _embed_seq(x, tau, n)
        M = len(Em)
        A = _np.tile(Em, (len(Em), 1, 1))
        B = _np.transpose(A, [1, 0, 2])
        square_dists = (A - B) ** 2  # square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
        D = _np.sqrt(square_dists[:, :, :].sum(axis=2))  # D[i,j] = ||Em[i]-Em[j]||_2

        # Exclude elements within T of the diagonal
        band = _np.tri(D.shape[0], k=T) - _np.tri(D.shape[0], k=-T - 1)
        band[band == 1] = _np.inf
        neighbors = (D + band).argmin(axis=0)  # nearest neighbors more than T steps away

        # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
        inc = _np.tile(_np.arange(M), (M, 1))
        row_inds = (_np.tile(_np.arange(M), (M, 1)).T + inc)
        col_inds = (_np.tile(neighbors, (M, 1)) + inc.T)
        in_bounds = _np.logical_and(row_inds <= M - 1, col_inds <= M - 1)
        # Uncomment for old (miscounted) version
        # in_bounds = numpy.logical_and(row_inds < M - 1, col_inds < M - 1)
        row_inds[~in_bounds] = 0
        col_inds[~in_bounds] = 0

        # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
        neighbor_dists = _np.ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)
        J = (~neighbor_dists.mask).sum(axis=1)  # number of in-bounds indices by row
        # Set invalid (zero) values to 1; log(1) = 0 so sum is unchanged

        neighbor_dists[neighbor_dists == 0] = 1

        # !!! this fixes the divide by zero in log error !!!
        neighbor_dists.data[neighbor_dists.data == 0] = 1

        d_ij = _np.sum(_np.log(neighbor_dists.data), axis=1)
        mean_d = d_ij[J > 0] / J[J > 0]

        x = _np.arange(len(mean_d))
        X = _np.vstack((x, _np.ones(len(mean_d)))).T
        [m, c] = _np.linalg.lstsq(X, mean_d, rcond=None)[0]
        Lexp = fs * m
        return Lexp
    tau = kwargs["Tau"]
    n = kwargs["n"]
    T = kwargs["T"]
    fs = kwargs["fs"]
    return _np.apply_along_axis(LLE_1d, axis, epochs, tau, n, T, fs)


def maximum(epochs, axis, **kwargs):
    return _np.max(epochs, axis=axis)


def mean(epochs, axis, **kwargs):
    return _np.mean(epochs, axis=axis)


def median(epochs, axis, **kwargs):
    return _np.median(epochs, axis=axis)


def minimum(epochs, axis, **kwargs):
    return _np.min(epochs, axis=axis)


def non_linear_energy(epochs, axis, **kwargs):
    return _np.apply_along_axis(lambda epoch: _np.mean((_np.square(epoch[1:-1]) - epoch[2:] * epoch[:-2])), axis, epochs)


def petrosian_fractal_dimension(epochs, axis, **kwargs):
    def pfd_1d(X, D=None):
        # taken from pyeeg
        """Compute Petrosian Fractal Dimension of a time series from either two
        cases below:
            1. X, the time series of type list (default)
            2. D, the first order differential sequence of X (if D is provided,
               recommended to speed up)

        In case 1, D is computed using Numpy's difference function.

        To speed up, it is recommended to compute D before calling this function
        because D may also be used by other functions whereas computing it here
        again will slow down.
        """
        if D is None:
            D = _np.diff(X)
            D = D.tolist()
        N_delta = 0  # number of sign changes in derivative of the signal
        for i in range(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(X)
        return _np.log10(n) / (_np.log10(n) + _np.log10(n / n + 0.4 * N_delta))
    return _np.apply_along_axis(pfd_1d, axis, epochs)


def skewness(epochs, axis, **kwargs):
    return _skew(epochs, axis=axis, bias=False)


def svd_entropy(epochs, axis, **kwargs):
    def svd_entropy_1d(X, Tau, DE, W):
        # taken from pyeeg
        """Compute SVD Entropy from either two cases below:
        1. a time series X, with lag tau and embedding dimension dE (default)
        2. a list, W, of normalized singular values of a matrix (if W is provided,
        recommend to speed up.)

        If W is None, the function will do as follows to prepare singular spectrum:

            First, computer an embedding matrix from X, Tau and DE using pyeeg
            function embed_seq():
                        M = embed_seq(X, Tau, DE)

            Second, use scipy.linalg function svd to decompose the embedding matrix
            M and obtain a list of singular values:
                        W = svd(M, compute_uv=0)

            At last, normalize W:
                        W /= sum(W)

        Notes
        -------------

        To speed up, it is recommended to compute W before calling this function
        because W may also be used by other functions whereas computing it here
        again will slow down.
        """

        if W is None:
            Y = _embed_seq(X, Tau, DE)
            W = _np.linalg.svd(Y, compute_uv=0)
            W /= sum(W)  # normalize singular values

        return -1 * sum(W * _np.log(W))
    Tau = kwargs["Tau"]
    DE = kwargs["DE"]
    W = kwargs["W"]
    return _np.apply_along_axis(svd_entropy_1d, axis, epochs, Tau, DE, W)


def zero_crossing(epochs, axis, **kwargs):
    e = 0.01
    norm = epochs - epochs.mean()
    return _np.apply_along_axis(lambda epoch: _np.sum((epoch[:-5] <= e) & (epoch[5:] > e)), axis, norm)


def zero_crossing_derivative(epochs, axis, **kwargs):
    e = 0.01
    diff = _np.diff(epochs)
    norm = diff-diff.mean()
    return _np.apply_along_axis(lambda epoch: _np.sum(((epoch[:-5] <= e) & (epoch[5:] > e))), axis, norm)
