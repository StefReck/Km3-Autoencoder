# -*- coding: utf-8 -*-
"""
Plot logfile made from calculate mse.
"""
import matplotlib.pyplot as plt

from plotting.plot_statistics import read_out_file

file="../results/data/mse__s.txt"
data_dict=read_out_file(file)

whats_there = ("MSE above3","MSE below","MSE")
labels = ("MSE of high count bins", "MSE of low count bins", "Total MSE")
colors = ("green","orange","blue")

fig, ax = plt.subplots()
plt.title("Constituents of the test MSE")
plt.grid()

ax.plot(data_dict["Epoch"], data_dict["MSE below"], "o-", label=labels[1], c=colors[1])
ax.plot(data_dict["Epoch"], data_dict["MSE"], "o-", label=labels[2], c=colors[2])
ax.set_xlabel("Autoencoder epoch")
ax.set_ylabel("Test MSE")
plt.legend(loc="upper left")

ax2 = ax.twinx()
ax2.plot(data_dict["Epoch"], data_dict["MSE above3"], "o-", label=labels[0], c=colors[0])
ax2.set_ylabel("Test MSE")

plt.legend(loc="upper right")
plt.show()
