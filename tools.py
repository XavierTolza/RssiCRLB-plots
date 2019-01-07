import numpy as np


def ellipse_from_covariance(cov, angle=None, n_sigma=2):
    if angle is None:
        angle = np.linspace(0, np.pi * 2, 100)
    L = np.linalg.cholesky(cov)
    C = np.array([np.cos(angle), np.sin(angle)])
    res = n_sigma * L.dot(C)
    return res


def plot_ellipse(cov, axe, x0, **kwargs):
    angle = np.linspace(0, 2 * np.pi, 100)
    el = ellipse_from_covariance(cov, angle) + x0[:, None]
    return axe.plot(*el, **kwargs)
