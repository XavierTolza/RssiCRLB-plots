import numpy as np
import matplotlib.pyplot as plt
from bokeh import palettes
from tools import plot_ellipse

colors = np.reshape(palettes.Category20[20], (-1, 2))

def plot_results(transmitters_coordinates, estimate, ap_coordinates, crlb):

    axe = plt.figure().add_subplot(111)

    # Drawing covariance ellipses
    mean_cov = [(np.mean(i, axis=0), np.cov(i.T)) for i in np.moveaxis(estimate, 1, 0)]
    for i, j, c,d in zip(mean_cov, crlb, colors,transmitters_coordinates):
        plot_ellipse(i[1], axe, i[0], color=c[1], ls=":")
        plot_ellipse(j, axe, d, color=c[0])

    for t,c in zip(range(len(transmitters_coordinates)),colors):
        axe.scatter(*estimate[:, t, :].T, label="Transmitter %i" % t, alpha=.2,color=c[1])
    axe.scatter(*ap_coordinates.T, color="k", label="Access points")

    axe.legend()
    return axe
    # plt.show()