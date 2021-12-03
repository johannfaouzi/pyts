"""
==========================
Singular Spectrum Analysis
==========================

Signals such as time series can be seen as a sum of different signals such
as trends and noise. Decomposing time series into several time series can
be useful in order to keep the most important information. One decomposition
algorithm is Singular Spectrum Analysis. This example illustrates the
decomposition of a time series into several subseries using this algorithm and
visualizes the different subseries extracted.
It is implemented as :class:`pyts.decomposition.SingularSpectrumAnalysis`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis
from pyts.datasets import make_cylinder_bell_funnel

# Parameters
n_samples, n_timestamps = 100, 128

X_cbf, y = make_cylinder_bell_funnel(n_samples=100)
X_period = 3*np.sin(np.arange(n_timestamps))

X = X_cbf[:, :n_timestamps] + X_period

# We decompose the time series into three subseries
window_size = 15
# Singular Spectrum Analysis
ssa = SingularSpectrumAnalysis(window_size=window_size, groups="auto")
X_ssa = ssa.fit_transform(X)

# Show the results for the first time series and its subseries
plt.figure(figsize=(16, 6))

ax1 = plt.subplot(121)
ax1.plot(X[0], 'o-', label='Original')
ax1.legend(loc='best', fontsize=14)
ax1.set_ylim([np.min(X[0])*1.1, np.max(X[0])*1.1])

ax2 = plt.subplot(122)
labels = ["trend", "periodic", "residual"]
for i in range(3):
    ax2.plot(X_ssa[0, i], 'o--', label=labels[i])
ax2.legend(loc='best', fontsize=14)
ax2.set_ylim([np.min(X[0])*1.1, np.max(X[0])*1.1])

plt.suptitle('Singular Spectrum Analysis', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
