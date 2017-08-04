from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pyts.utils import paa, sax, gaf, mtf, dtw, fast_dtw


def plot_ts(ts, output_file=None, **kwargs):
    """Plot the time series.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    output_file : str or None (default = None)
        if str, save the figure.

    kwargs : keyword arguments
        kwargs for matplotlib.pyplot.plot
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    plt.plot(ts, color='#7f7f7f', **kwargs)
    if output_file is not None:
        plt.savefig(output_file)


def plot_standardscaler(ts, epsilon=1e-3, output_file=None, **kwargs):
    """Plot the original and standardized time series.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    epsilon : float (default = 1e-3)
        value added to empirical variance before dividing.

    output_file : str or None (default = None)
        if str, save the figure.

    kwargs : keyword arguments
        kwargs for matplotlib.pyplot.plot
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    # Check parameters
    if not isinstance(epsilon, (int, float)):
        raise TypeError("'epsilon' must be a float or an int.")
    if epsilon < 0:
        raise ValueError("'epsilon' must be greater or equal than 0.")

    ts_standardscaler = (ts - ts.mean()) / (ts.std() + epsilon)

    plt.plot(ts, color='#7f7f7f', label='Original', **kwargs)
    plt.plot(ts_standardscaler, color='#1f77b4', label='Standardized', **kwargs)
    plt.legend(loc='best')

    if output_file is not None:
        plt.savefig(output_file)


def plot_paa(ts, window_size=None, output_size=None, overlapping=True, output_file=None, **kwargs):
    """Plot the time series before and after PAA transformation.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    window_size : int or None (default = None)
        size of the sliding window

    output_size : int or None (default = None)
        size of the returned time series

    overlapping : bool (default = True)
        when output_size is specified, the window_size is fixed
        if overlapping is True and may vary if overlapping is False.
        Ignored if window_size is specified.

    output_file : str or None (default = None)
        if str, save the figure.

    kwargs : keyword arguments
        kwargs for matplotlib.pyplot.plot
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    # Size of ts
    ts_size = ts.size

    # Check parameters and compute window_size if output_size is given
    if (window_size is None and output_size is None):
        raise ValueError("'window_size' xor 'output_size' must be specified.")
    elif (window_size is not None and output_size is not None):
        raise ValueError("'window_size' xor 'output_size' must be specified.")
    elif (window_size is not None and output_size is None):
        if not isinstance(overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if not isinstance(window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if window_size < 1:
            raise ValueError("'window_size' must be greater or equal than 1.")
        if window_size > ts_size:
            raise ValueError("'window_size' must be lower or equal than the size of each time series.")
    else:
        if not isinstance(overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if not isinstance(output_size, int):
            raise TypeError("'output_size' must be an integer.")
        if output_size < 1:
            raise ValueError("'output_size' must be greater or equal than 1.")
        if output_size > ts_size:
            raise ValueError("'output_size' must be lower or equal than the size of each time series.")
        window_size = ts_size // output_size
        window_size += 0 if ts_size % output_size == 0 else 1

    indices, mean = paa(ts, ts.size, window_size, overlapping, output_size, plot=True)
    indices_len = len(indices)

    plt.plot(ts, color='#1f77b4', **kwargs)
    for i in range(indices_len):
        plt.plot(indices[i], np.repeat(mean[i], indices[i].size), 'r-')

    plt.axvline(x=indices[0][0], ls='--', linewidth=1, color='k')
    for i in range(indices_len - 1):
        plt.axvline(x=(indices[i][-1] + indices[i + 1][0]) / 2, ls='--', linewidth=1, color='k')
    plt.axvline(x=indices[indices_len - 1][-1], ls='--', linewidth=1, color='k')

    if output_file is not None:
        plt.savefig(output_file)


def plot_paa_sax(
        ts, window_size=None, output_size=None, overlapping=True,
        n_bins=8, quantiles='gaussian', output_file=None, **kwargs):

    """Plot the original time series, the time series after PAA
    transformation and the time series after PAA and SAX transformations.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    window_size : int or None (default = None)
        size of the sliding window

    output_size : int or None (default = None)
        size of the returned time series

    overlapping : bool (default = True)
        when output_size is specified, the window_size is fixed
        if overlapping is True and may vary if overlapping is False.
        Ignored if window_size is specified.

    n_bins : int (default = 8)
        number of bins (also known as the size of the alphabet)

    quantiles : str (default = 'gaussian')
        the way to compute quantiles. Possible values:

            - 'gaussian' : quantiles from a gaussian distribution N(0,1)
            - 'empirical' : empirical quantiles

    output_file : str or None (default = None)
        if str, save the figure.

    kwargs : keyword arguments
        kwargs for matplotlib.pyplot.plot
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    # Size of ts
    ts_size = ts.size

    # Check parameters for PAA and compute window_size if output_size is given
    if (window_size is None and output_size is None):
        raise ValueError("'window_size' xor 'output_size' must be specified.")
    elif (window_size is not None and output_size is not None):
        raise ValueError("'window_size' xor 'output_size' must be specified.")
    elif (window_size is not None and output_size is None):
        if not isinstance(overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if not isinstance(window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if window_size < 1:
            raise ValueError("'window_size' must be greater or equal than 1.")
        if window_size > ts_size:
            raise ValueError("'window_size' must be lower or equal than the size 'ts'.")
    else:
        if not isinstance(overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if not isinstance(output_size, int):
            raise TypeError("'output_size' must be an integer.")
        if output_size < 1:
            raise ValueError("'output_size' must be greater or equal than 1.")
        if output_size > ts_size:
            raise ValueError("'output_size' must be lower or equal than the size of 'ts'.")
        window_size = ts_size // output_size
        window_size += 0 if ts_size % output_size == 0 else 1

    # Check parameters for SAX
    if not isinstance(n_bins, int):
        raise TypeError("'n_bins' must be an integer")
    if n_bins < 2:
        raise ValueError("'n_bins' must be greater or equal than 2")
    if n_bins > 52:
        raise ValueError("'n_bins' must be lower or equal than 52")
    if quantiles not in ['gaussian', 'empirical']:
        raise ValueError("'quantiles' must be either 'gaussian' or 'empirical'")

    indices, ts_paa = paa(ts, ts.size, window_size, overlapping, output_size, plot=True)
    indices_len = len(indices)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(ts, color='#1f77b4', **kwargs)
    for i in range(indices_len):
        plt.plot(indices[i], np.repeat(ts_paa[i], indices[i].size), 'r-')

    plt.axvline(x=indices[0][0], ls='--', linewidth=1, color='k')
    for i in range(indices_len - 1):
        plt.axvline(x=(indices[i][-1] + indices[i + 1][0]) / 2, ls='--', linewidth=1, color='k')
    plt.axvline(x=indices[indices_len - 1][-1], ls='--', linewidth=1, color='k')

    # Alphabet
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Compute gaussian quantiles if quantiles == 'gaussian'
    if quantiles == 'gaussian':
        quantiles = scipy.stats.norm.ppf(np.linspace(0, 1, num=n_bins + 1)[1:])
        ts_sax = sax(ts_paa, n_bins, quantiles, alphabet, plot=True)
    else:
        quantiles, ts_sax = sax(ts_paa, n_bins, quantiles, alphabet, plot=True)

    for i in range(n_bins - 1):
        plt.axhline(y=quantiles[i], ls='--', lw=1, color='g')

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    for i in range(indices_len):
        x_pos = (np.percentile(indices[i], [50]) - x_lim[0]) / (x_lim[1] - x_lim[0])
        y_pos = (ts_paa[i] - y_lim[0]) / (y_lim[1] - y_lim[0])
        ax.text(x_pos, y_pos, ts_sax[i],
                horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes, color='m', fontsize=25)

    if output_file is not None:
        plt.savefig(output_file)


def plot_sax(ts, n_bins, quantiles='gaussian', output_file=None, **kwargs):
    """Plot the time series before and after SAX transformation.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    n_bins : int (default = 8)
        number of bins (also known as the size of the alphabet)

    quantiles : str (default = 'gaussian')
        the way to compute quantiles. Possible values:

            - 'gaussian' : quantiles from a gaussian distribution N(0,1)
            - 'empirical' : empirical quantiles

    output_file : str or None (default = None)
        if str, save the figure.

    kwargs : keyword arguments
        kwargs for matplotlib.pyplot.plot
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    # Check parameters
    if not isinstance(n_bins, int):
        raise TypeError("'n_bins' must be an integer.")
    if n_bins < 2:
        raise ValueError("'n_bins' must be greater or equal than 2.")
    if n_bins > 52:
        raise ValueError("'n_bins' must be lower or equal than 52.")
    if quantiles not in ['gaussian', 'empirical']:
        raise ValueError("'quantiles' must be either 'gaussian' or 'empirical'.")

    # Alphabet
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Compute gaussian quantiles if quantiles == 'gaussian'
    if quantiles == 'gaussian':
        quantiles = scipy.stats.norm.ppf(np.linspace(0, 1, num=n_bins + 1)[1:])
        ts_sax = sax(ts, n_bins, quantiles, alphabet, plot=True)
    else:
        quantiles, ts_sax = sax(ts, n_bins, quantiles, alphabet, plot=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(ts, color='r', **kwargs)

    for i in range(n_bins - 1):
        plt.axhline(y=quantiles[i], ls='--', lw=1, color='g')

    if 1:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        for i in range(len(ts_sax)):
            x_pos = (i - x_lim[0]) / (x_lim[1] - x_lim[0])
            y_pos = (ts[i] - y_lim[0]) / (y_lim[1] - y_lim[0])
            ax.text(x_pos, y_pos, ts_sax[i],
                    horizontalalignment='center', verticalalignment='bottom',
                    transform=ax.transAxes, color='m', fontsize=25)

    if output_file is not None:
        plt.savefig(output_file)


def plot_dtw(x, y, dist='absolute', output_file=None):
    """Plot the optimal warping path between two time series.

    Parameters
    ----------
    x : np.array, shape = [n_features]
        first time series

    y : np.array, shape = [n_features]
        first time series

    dist : str or callable (default = 'absolute')
        cost distance between two real numbers. Possible values:

        - 'absolute' : absolute value of the difference
        - 'square' : square of the difference
        - callable : first two parameters must be real numbers
            and it must return a real number.

    output_file : str or None (default = None)
        if str, save the figure.
    """

    # Check input data
    if not (isinstance(x, np.ndarray) and x.ndim == 1):
        raise ValueError("'x' must be a 1-dimensional np.ndarray.")
    if not (isinstance(y, np.ndarray) and y.ndim == 1):
        raise ValueError("'y' must be a 1-dimensional np.ndarray.")
    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")

    # Size of x
    x_size = x.size

    # Check parameters
    if not (callable(dist) or dist in ['absolute', 'square']):
        raise ValueError("'dist' must be a callable or 'absolute' or 'square'.")

    D, path = dtw(x, y, dist=dist, return_path=True)

    x_1 = np.arange(x_size + 1)
    z_1 = np.zeros([x_size + 1, x_size + 1])
    for i in range(len(path)):
        z_1[path[i][0], path[i][1]] = 1

    plt.pcolor(x_1, x_1, z_1, edgecolors='k', cmap='Greys')
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)

    if output_file is not None:
        plt.savefig(output_file)


def plot_fastdtw(x, y, window_size, dist='absolute', output_file=None):
    """Plot the optimal warping path between two time series subject to
    a constraint region.

    Parameters
    ----------
    x : np.array, shape = [n_features]
        first time series

    y : np.array, shape = [n_features]
        first time series

    window_size : int
        size of the window for PAA

    dist : str or callable (default = 'absolute')
        cost distance between two real numbers. Possible values:

        - 'absolute' : absolute value of the difference
        - 'square' : square of the difference
        - callable : first two parameters must be real numbers
            and it must return a real number.

    output_file : str or None (default = None)
        if str, save the figure.
    """

    # Check input data
    if not (isinstance(x, np.ndarray) and x.ndim == 1):
        raise ValueError("'x' must be a 1-dimensional np.ndarray.")
    if not (isinstance(y, np.ndarray) and y.ndim == 1):
        raise ValueError("'y' must be a 1-dimensional np.ndarray.")
    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same size.")

    # Size of x
    x_size = x.size

    # Check parameters
    if not isinstance(window_size, int):
        raise TypeError("'window_size' must be an integer.")
    if window_size < 1:
        raise ValueError("'window_size' must be greater or equal than 1.")
    if window_size > x_size:
        raise ValueError("'window_size' must be lower or equal than the size 'x'.")
    if not (callable(dist) or dist in ['absolute', 'square']):
        raise ValueError("'dist' must be a callable or 'absolute' or 'square'.")

    region, D, path = fast_dtw(x, y, window_size=window_size, approximation=False,
                               dist=dist, return_path=True)

    x_1 = np.arange(x_size + 1)

    z_1 = np.zeros([x_size + 1, x_size + 1])
    for i in range(x_size):
        for j in region[i]:
            z_1[j, i] = 0.5

    for i in range(len(path)):
        z_1[path[i][0], path[i][1]] = 1

    plt.pcolor(x_1, x_1, z_1, edgecolors='k', cmap='Greys')
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)

    if output_file is not None:
        plt.savefig(output_file)


def plot_gasf(ts, image_size=32, overlapping=False, scale='-1',
              cmap='rainbow', output_file=None):
    """Plot the image obtained after GASF transformation.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    image_size : int (default = 32)
        determines the shapes of the output image :
        image_size x image_size

    overlapping : bool (default = False)
        if True, reducing the size of the time series with PAA is
        done with possible overlapping windows.

    scale : str (default = '-1')
        the lower bound of the scaled time series. Possible values:

            - '-1' : the time series are scaled in [-1,1]
            - '0' : the time series are scaled in [0,1]

    output_file : str or None (default = None)
        if str, save the figure.
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    # Size of ts
    ts_size = ts.size

    # Check parameters
    if not isinstance(image_size, int):
        raise TypeError("'image_size' must be an integer.")
    if image_size < 2:
        raise ValueError("'image_size' must be greater or equal than 2.")
    if image_size > ts_size:
        raise ValueError("'image_size' must be lower or equal than the size of 'ts'.")
    if not isinstance(overlapping, (float, int)):
        raise TypeError("'overlapping' must be a boolean.")
    if scale not in ['0', '-1']:
        raise ValueError("'scale' must be either '0' or '-1'.")

    image_gasf = gaf(ts, ts_size, image_size, overlapping, 's', scale)

    plt.imshow(image_gasf, cmap=cmap)
    plt.axis('off')

    if output_file is not None:
        plt.savefig(output_file)


def plot_gadf(ts, image_size, overlapping=False, scale='-1',
              cmap='rainbow', output_file=None):
    """Plot the image obtained after GADF transformation.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    image_size : int (default = 32)
        determines the shapes of the output image :
        image_size x image_size

    overlapping : bool (default = False)
        if True, reducing the size of the time series with PAA is
        done with possible overlapping windows.

    scale : str (default = '-1')
        the lower bound of the scaled time series. Possible values:

            - '-1' : the time series are scaled in [-1,1]
            - '0' : the time series are scaled in [0,1]

    output_file : str or None (default = None)
        if str, save the figure.
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    # Size of ts
    ts_size = ts.size

    # Check parameters
    if not isinstance(image_size, int):
        raise TypeError("'image_size' must be an integer.")
    if image_size < 2:
        raise ValueError("'image_size' must be greater or equal than 2.")
    if image_size > ts_size:
        raise ValueError("'image_size' must be lower or equal than the size of 'ts'.")
    if not isinstance(overlapping, (float, int)):
        raise TypeError("'overlapping' must be a boolean.")
    if scale not in ['0', '-1']:
        raise ValueError("'scale' must be either '0' or '-1'.")

    image_gadf = gaf(ts, ts_size, image_size, overlapping, 'd', scale)

    plt.imshow(image_gadf, cmap=cmap)
    plt.axis('off')

    if output_file is not None:
        plt.savefig(output_file)


def plot_mtf(ts, image_size=32, n_bins=8, quantiles='empirical',
             overlapping=False, cmap='rainbow', output_file=None):
    """Plot the image obtained after MTF transformation.

    Parameters
    ----------
    ts : np.array, shape = [n_features]
        time series to plot

    image_size : int (default = 32)
        determines the shapes of the output image :
        image_size x image_size

    n_bins : int (default = 8)
        number of bins (also known as the size of the alphabet)

    quantiles : str (default = 'gaussian')
        the way to compute quantiles. Possible values:

            - 'gaussian' : quantiles from a gaussian distribution N(0,1)
            - 'empirical' : empirical quantiles

    overlapping : bool (default = False)
        if False, reducing the image with the blurring kernel
        will be applied on non-overlapping rectangles; if True,
        it will be applied on possible overlapping squares.

    cmap : str
        color map from matplotlib.pyplot

    output_file : str or None (default = None)
        if str, save the figure.
    """

    # Check input data
    if not (isinstance(ts, np.ndarray) and ts.ndim == 1):
        raise ValueError("'ts' must be a 1-dimensional np.ndarray.")

    # Size of ts
    ts_size = ts.size

    # Check parameters
    if not isinstance(image_size, int):
        raise TypeError("'size' must be an integer.")
    if image_size < 2:
        raise ValueError("'image_size' must be greater or equal than 2.")
    if image_size > ts_size:
        raise ValueError("'image_size' must be lower or equal than the size of 'ts'.")
    if not isinstance(n_bins, int):
        raise TypeError("'n_bins' must be an integer.")
    if n_bins < 2:
        raise ValueError("'n_bins' must be greater or equal than 2.")
    if quantiles not in ['gaussian', 'empirical']:
        raise ValueError("'quantiles' must be either 'gaussian' or 'empirical'.")
    if not isinstance(overlapping, (float, int)):
        raise TypeError("'overlapping' must be a boolean.")

    # Compute gaussian quantiles if quantiles == 'gaussian'
    if quantiles == 'gaussian':
        quantiles = scipy.stats.norm.ppf(np.linspace(0, 1, num=n_bins + 1)[1:])

    image_mtf = mtf(ts, ts_size, image_size, n_bins, quantiles, overlapping)

    plt.imshow(image_mtf, cmap=cmap)
    plt.axis('off')

    if output_file is not None:
        plt.savefig(output_file)
