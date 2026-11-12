"""Ordinary least squares of y on x, one residue per row.

    fit = fit_ols(x, y)        y ~ fit["intercept"] + x @ fit["coefficients"]
    predict(fit, x)

The direction is the choice of the caller. With h the residual stream and v the variables of a residue (e.g. the cos and sin of an angle):
    encoder = fit_ols(v, h)    variables -> activation. Its coefficients are the vectors for steering and ablation.
    decoder = fit_ols(h, v)    activation -> variables. For prediction only: its coefficients also cancel the other variance of h.
"""

import numpy as np


def fit_ols(x, y, chunk=100_000):
    """Least squares of y (n, q) on [1, x], x (n, p): {"intercept": (q,), "coefficients": (p, q)}.
    x and y are read `chunk` rows at a time, as float64, so they can be float16 arrays on disk (np.load(..., mmap_mode="r"))."""
    n, p, q = len(x), x.shape[1], y.shape[1]
    sum_x, sum_y, sum_xx, sum_xy = np.zeros(p), np.zeros(q), np.zeros((p, p)), np.zeros((p, q))
    for start in range(0, n, chunk):
        x_rows = np.asarray(x[start:start + chunk], dtype=np.float64)
        y_rows = np.asarray(y[start:start + chunk], dtype=np.float64)
        sum_x += x_rows.sum(0)
        sum_y += y_rows.sum(0)
        sum_xx += x_rows.T @ x_rows
        sum_xy += x_rows.T @ y_rows
    mean_x = sum_x / n
    mean_y = sum_y / n
    cov_xx = sum_xx / n - np.outer(mean_x, mean_x)  # (p, p)
    cov_xy = sum_xy / n - np.outer(mean_x, mean_y)  # (p, q)
    coefficients = np.linalg.solve(cov_xx, cov_xy)
    intercept = mean_y - mean_x @ coefficients
    return {"intercept": intercept, "coefficients": coefficients}


def predict(fit, x):
    """The fitted y (n, q) of x (n, p)."""
    return fit["intercept"] + x @ fit["coefficients"]
