import numpy as np
import matplotlib.pyplot as plt
from bokeh import palettes
from bokeh.layouts import row
from bokeh.models import FixedTicker
from bokeh.plotting import figure
from bokeh.sphinxext.sample import Bar

from tools import plot_ellipse

colors = palettes.Greys[6]


def plot_results(transmitters_coordinates, estimate, ap_coordinates, crlb, **kwargs):
    p1 = figure(title="2-Sigma covariance plot", x_axis_label="X_coordinates", y_axis_label="Y coordinates",
                active_scroll="wheel_zoom", plot_width=800, **kwargs)

    p2 = figure(title="RMSE vs CRLB", x_axis_label="Transmitter", y_axis_label="RMSE",
                active_scroll="wheel_zoom", plot_width=300,**kwargs)

    # Drawing covariance ellipses
    mean_cov = [(np.mean(i, axis=0), np.cov(i.T)) for i in np.moveaxis(estimate, 1, 0)]
    for i, j, c, d in zip(mean_cov, crlb, colors, transmitters_coordinates):
        plot_ellipse(i[1], p1, i[0], line_color="black", line_dash="dashed", legend="Simulation", line_width=1.5)
        plot_ellipse(j, p1, d, line_color="black", line_dash="solid", legend="CRLB", line_width=2)

    for t, c in zip(range(len(transmitters_coordinates)), "circle,x,square".split(",")):
        getattr(p1, c)(*estimate[::2, t, :].T, legend="Transmitter %i estimates" % t, alpha=.5, fill_color="grey",
                       line_color="grey",line_width=2)
    p1.circle(*ap_coordinates.T, line_color="black", legend="Access points", fill_color="black",size=10)

    RMSE = np.mean(np.linalg.norm(estimate - transmitters_coordinates[None, :, :], axis=2), axis=0)
    x = np.arange(len(transmitters_coordinates))
    p2.vbar(x=x, top=RMSE, legend="RSME (Simulation)", width=.45, fill_color="black", line_color=None)
    p2.vbar(x=x + .45, top=[np.trace(i) for i in crlb], legend="CRLB", width=.45, fill_color="grey", line_color=None)

    p2.legend.location = "bottom_right"
    p1.legend.location = "center_right"
    p2.xaxis.ticker = FixedTicker(ticks=[0,1])
    return row(p1, p2)
    # plt.show()
