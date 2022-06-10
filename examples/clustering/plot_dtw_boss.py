"""
=========================================
Time Series Clustering with DTW and BOSS
=========================================

This example shows the differences between various metrics
related to time series clustering. Besides the Euclidean distance,
:func:`pyts.metrics.dtw` and :func:`pyts.metrics.boss` are considered to
analyze the :func:`pyts.datasets.make_cylinder_bell_funnel` dataset.
While the Euclidean distance and DTW are locally sensitive, clustering
with BOSS remains almost unchanged by a time shift. Thus, it depends strongly
on the particular time series which metric is the more accurate one.

The example is inspired by

[1] P. Schäfer, "The BOSS is concerned with time series classification
    in the presence of noise". Data Mining and Knowledge Discovery,
    29(6), 1505-1530 (2015).
"""

# Author: Lucas Plagwitz <lucas.plagwitz@uni-muenster.de>
# License: BSD-3-Clause

import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


from pyts.metrics import dtw, boss
from pyts.transformation import BOSS
from pyts.datasets import make_cylinder_bell_funnel


def create_dist_matrix(dataset, dist_func, **kwargs):
    distance_mat = np.zeros((len(dataset), len(dataset)))
    for i, j in itertools.product(range(len(dataset)),
                                  range(len(dataset))):
        distance_mat[i, j] = dist_func(dataset[i], dataset[j], **kwargs)
    return distance_mat


def plot_dendrogram(model, **kwargs):
    # function copied from sklearn:
    # plot_agglomerative_dendrogram.html
    #
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


n_samples = 14
X, y = make_cylinder_bell_funnel(n_samples=n_samples, random_state=42,
                                 shuffle=False)


fig, axes = plt.subplots(2, 3, figsize=(16, 12))
axes = np.ravel(axes)

k_axis = 0
for time_shift in [False, True]:
    if time_shift:
        for i in range(n_samples):
            X[i] = np.roll(X[i], np.arange(0, 50, 10)[i % 5])

    for metric in ["euc", "dtw", "boss"]:
        if metric == "dtw":
            dist_mat = create_dist_matrix(X, dtw, method="sakoechiba",
                                          options={"window_size": 15})
        elif metric == "boss":
            dist_mat = create_dist_matrix(BOSS(sparse=False, n_bins=3,
                                               word_size=3).fit_transform(X),
                                          boss)
        else:
            dist_mat = create_dist_matrix(X, euclidean)

        model = AgglomerativeClustering(distance_threshold=0,
                                        n_clusters=None,
                                        affinity="precomputed",
                                        linkage="complete")
        cluster = model.fit_predict(dist_mat)

        plot_dendrogram(model, orientation='left',
                        ax=axes[k_axis], labels=y)

        if k_axis == 0:
            axes[k_axis].set_ylabel("normal", rotation=90,
                                    size='xx-large')
            axes[k_axis].set_title("Euclidean", size='xx-large')
        if k_axis == 1:
            axes[k_axis].set_title("DTW", size='xx-large')
        if k_axis == 2:
            axes[k_axis].set_title("BOSS", size='xx-large')
        if k_axis == 3:
            axes[k_axis].set_ylabel("time shifted", rotation=90,
                                    size='xx-large')
        k_axis += 1

plt.show()
