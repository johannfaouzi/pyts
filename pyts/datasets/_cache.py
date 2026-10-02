"""Default cache directory for datasets downloaded from an online archive."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import os

from platformdirs import user_cache_dir


def _default_data_home(archive: str) -> str:
    """Default download directory for the ``fetch_*_dataset`` functions.

    Resolves to the OS-appropriate per-user cache directory (e.g.
    ``~/Library/Caches/pyts`` on macOS, ``~/.cache/pyts`` -- or
    ``$XDG_CACHE_HOME/pyts`` if set -- on Linux, and
    ``%LOCALAPPDATA%\\pyts\\Cache`` on Windows), not the installed
    package's own directory: the latter may not be writable (e.g. a
    system-wide install) and downloaded archives have nothing to do with
    pyts's own source/package files.

    Parameters
    ----------
    archive : str
        Name of the archive ('UCR' or 'UEA').

    Returns
    -------
    path : str
        The default directory for this archive's downloaded datasets.

    """
    return os.path.join(user_cache_dir('pyts'), archive)
