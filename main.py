import json
import os

import pandas
from pandas import DataFrame

csv_name = "ffly_reu_measurements_auto_placement.csv"

def convert_measurements():
    path = "/media/xavier/Data/Code/Pycharm/indoor_loc/Data/Measurements/"
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
    df = pandas.read_csv(csv_name)

    for mode, df_mode in df.groupby("placement_mode"):
        for


if __name__ == '__main__':
    convert_measurements()