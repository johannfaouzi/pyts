"""Parser for the ``.ts`` time series file format.

As of 2026, `timeseriesclassification.com <https://timeseriesclassification.com>`_
serves the UCR and UEA archives through the ``aeon-toolkit`` endpoint, which
packages every dataset as a pair of ``{name}_TRAIN.ts``/``{name}_TEST.ts``
files rather than the plain-text/ARFF files used by the historical
``ClassificationDownloads`` endpoint. See the `aeon documentation
<https://www.aeon-toolkit.org/en/stable/api_reference/file_specifications/ts.html>`_
for the full format specification; only the subset of it needed by pyts
(comment header, ``@data`` section, ``:``-separated dimensions, ``,``-separated
observations, ``?``/``NaN`` missing values) is implemented here.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import numpy.typing as npt


def parse_ts_file(
    path: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.str_], str]:
    """Parse a ``.ts`` file into time series, labels and a description.

    Parameters
    ----------
    path : str
        Path to the ``.ts`` file.

    Returns
    -------
    X : array, shape = (n_samples, n_channels, n_timestamps)
        The time series. ``n_timestamps`` is the length of the longest
        series in the file; shorter series (when the problem is not of
        equal length) are padded with NaN's.

    y : array, shape = (n_samples,)
        The raw (string) class label for each sample.

    description : str
        The leading ``#``-prefixed comment lines of the file, which is
        where the archive stores the dataset description.

    """
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()

    description_lines = []
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            description_lines.append(stripped.lstrip('#').strip())
        if stripped.lower() == '@data':
            data_start = i + 1
            break
    if data_start is None:
        raise OSError(f"No '@data' tag found in {path}.")
    description = '\n'.join(description_lines)

    samples: list[list[list[float]]] = []
    labels = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        line = line.replace('?', 'nan')
        channels_raw = line.split(':')
        labels.append(channels_raw[-1].strip())
        channels = [
            [float(value) for value in channel.split(',') if value != '']
            for channel in channels_raw[:-1]
        ]
        samples.append(channels)

    n_samples = len(samples)
    n_channels = len(samples[0])
    max_length = max(len(channel) for sample in samples for channel in sample)

    X = np.full((n_samples, n_channels, max_length), np.nan)
    for i, sample in enumerate(samples):
        for j, channel in enumerate(sample):
            X[i, j, : len(channel)] = channel

    y = np.asarray(labels)

    return X, y, description
