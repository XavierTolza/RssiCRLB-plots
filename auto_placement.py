import json
import os

import pandas
from bokeh import events
from bokeh.models import CustomJS
from pandas import DataFrame
from bokeh.plotting import figure, output_file, show
import numpy as np

csv_name = "ffly_reu_measurements_auto_placement.csv"

def convert_measurements():
    path = "Measurements/"
    files = [i for i in os.listdir(path) if "ffly_reu" in i]
    jsons = {}
    for file in files:
        path_file = path + file
        with open(path_file,"r") as file_ptr:
            j = json.load(file_ptr)
        jsons[file] = j["multiple_rssi_measurements"]

    data_frame = DataFrame()
    for file, j in jsons.items():
        for pos_dict in j:
            x,y, ap_index, value = pos_dict["x"], pos_dict["y"], pos_dict["sources"], pos_dict["values"]
            df = DataFrame(dict(x = x, y = y, access_point_index=ap_index,rssi = value))
            if "optimal" in file:
                placement_mode="optimal"
                n_ap = int(file.split("optimal")[1][0])
            else:
                placement_mode = "non optimal"
                n_ap = 7
            df["placement_mode"] = placement_mode
            df["n_access_points"] = n_ap
            data_frame = data_frame.append(df)
    data_frame.to_csv(csv_name)


def plot_measurements():
    measures = ["non optimal"]+["%i mean"%i for i in np.arange(3,8)]
    types = "real,cramer,simu".split(",")
    root = "exported_data"

    output_file("plot.html",title="Publication 2 plot")
    p = figure(title="Cumulative density function of positionning error", x_axis_label='Distance in meters',
               y_axis_label='CDF',sizing_mode="stretch_both")

    import bokeh.palettes as palettes
    colors = np.reshape(palettes.Category20b[20]+palettes.Category20c[20][4:],(-1,4))

    def plot(x, y, muted, color_index, color_subindex, legend,lw=2,ls="solid"):
        l = p.line(x, y, legend=legend, line_width=lw, color=colors[color_index,color_subindex],
                   muted_alpha=0,line_dash=ls)
        l.muted=muted


    for measure_i, measure in enumerate(measures):
        for typ_i, (typ,lw,ls) in enumerate(zip(types,[2,1.5,1.5],"solid,dashed,dotted".split(","))):
            filename = "%s/%s_%s_cep.csv" % (root,measure,typ)
            _,x,y = pandas.read_csv(filename).values.transpose()
            plot(y,x,measure_i>0,measure_i,typ_i,"%s %s" % (measure.replace("mean","AP").replace("non","7 AP non"), typ),lw=lw,ls=ls)
    filename = "%s/random_cep.csv" % root
    _,x,y = pandas.read_csv(filename).values.transpose()
    plot(x,y,True,measure_i+1,0,"random")

    p.legend.location = "bottom_right"
    p.legend.click_policy="mute"
    def show_hide_legend(legend=p.legend[0]):
        legend.visible = not legend.visible
    p.js_on_event(events.DoubleTap, CustomJS.from_py_func(show_hide_legend))
    show(p)



if __name__ == '__main__':
    plot_measurements()