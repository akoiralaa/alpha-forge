
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def _get_quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int, copy=False)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()

def _recursive_bisect(
    cov: pd.DataFrame,
    sort_ix: list[int],
) -> pd.Series:
    weights = pd.Series(1.0, index=sort_ix)
    cluster_items = [sort_ix]

    while len(cluster_items) > 0:
        new_items = []
        for subset in cluster_items:
            if len(subset) <= 1:
                continue
            mid = len(subset) // 2
            left = subset[:mid]
            right = subset[mid:]

            # Compute cluster variance
            left_cov = cov.iloc[left, left]
            right_cov = cov.iloc[right, right]

            left_var = _cluster_var(left_cov)
            right_var = _cluster_var(right_cov)

            # Allocate inversely proportional to variance
            alpha = 1.0 - left_var / (left_var + right_var + 1e-10)
            weights[left] *= alpha
            weights[right] *= (1.0 - alpha)

            new_items.append(left)
            new_items.append(right)

        cluster_items = [x for x in new_items if len(x) > 1]

    return weights

def _cluster_var(cov: pd.DataFrame) -> float:
    n = len(cov)
    if n == 0:
        return 0.0
    w = np.ones(n) / n
    return float(w @ cov.values @ w)

def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    if returns.shape[1] == 1:
        return pd.Series([1.0], index=returns.columns)

    cov = returns.cov()
    corr = returns.corr()

    # Step 1: correlation distance matrix
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist.values, 0.0)

    # Convert to condensed form for linkage
    condensed = squareform(dist.values, checks=False)
    link = sch.linkage(condensed, method="ward")

    # Step 2: quasi-diagonalization
    sort_ix = _get_quasi_diag(link)
    sort_ix = [int(x) for x in sort_ix]

    # Step 3: recursive bisection
    weights = _recursive_bisect(cov, sort_ix)

    # Map back to column names
    result = pd.Series(0.0, index=returns.columns)
    for idx_pos, col_idx in enumerate(sort_ix):
        result.iloc[col_idx] = weights[col_idx]

    return result / result.sum()
