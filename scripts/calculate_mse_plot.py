# -*- coding: utf-8 -*-
"""
Plot logfile made from calculate mse.
"""
import matplotlib.pyplot as plt

from scripts.plotting.plot_statistics import read_out_file

file="results/data/mse__s.txt"
data_dict=read_out_file(file)

whats_there = ("MSE above3","MSE below","MSE")

for keyword in whats_there:
    plt.plot(data_dict["Epoch"], data_dict[whats_there])
