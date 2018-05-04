# -*- coding: utf-8 -*-
"""
Makes the plot for the unfreeze test.
"""
import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_statistics import read_out_file

#encoder_file="models/vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_test.txt"
debug=True

if debug:
    logfile_path="trained_vgg_5_200-unfreeze_autoencoder_epoch1_unfreeze_broken4_log.txt"
    save_as="../results/plots/unfreeze/broken4_vgg5_200_robustness.pdf"
else:
    home="/home/woody/capn/mppi013h/Km3-Autoencoder/"
    logfile_path=home+"models/vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_unfreeze_broken4_log.txt"
    save_as=home+"results/plots/unfreeze/broken4_vgg5_200_robustness.pdf"

data = read_out_file(logfile_path)
#This is present in the datafile, in addition to "epoch"
keys_robust = ['(Sim-Meas)', '(Upperlim-']
labels_robust=["Change to 'measured' data", "Change to 'upper limit'"]
colors_robust=["yellowgreen", "darkolivegreen"]

keys_acc = ['acc_sim', 'acc_meas', 'acc_ulim' ]
labels_acc=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
colors_acc=["orange", "blue", "navy"]



fig, ax = plt.subplots(figsize=(9,6))
ax2 = ax.twinx()
ax3=ax.twiny()

ax.set_xlim((0,46))
ax.set_xticks(np.arange(0,50,5))

ax.set_ylim(-10.5,10.5)
ax2.set_ylim(-1,41)

ax3.set_xlim((0,46))
ax3.set_xticks(np.arange(0,45,5)+1)
ax3.set_xticklabels([str(n) for n in range(9)])

ax3.set_xlabel("Unfrozen convolutional layers")

ax.set_xlabel("Epoch")
ax.set_ylabel("Absolute change to original accuracy (%)")
ax2.set_ylabel("Relative decrease changing datasets (%)")

ax.xaxis.grid()
ax.axhline(0, c="lightgrey", zorder=-1000, lw=0.5)

for no,key in enumerate(keys_acc):
    ydata = [100*(float(point)-float(data[key][0])) for point in data[key]]
    ax.plot(data["epoch"], ydata, "o-", label=labels_acc[no], color=colors_acc[no])
ax.legend(loc="upper left")


for no,key in enumerate(keys_robust):
    ax2.plot(data["epoch"], data[key], "o--", label=labels_robust[no], color=colors_robust[no])
ax2.legend(loc="upper right")

fig.subplots_adjust(top=0.85)
fig.suptitle("Robustness during successive unfreezing of encoder layers")

if save_as is not None:
    print("Saving plot to", save_as)
    plt.savefig(save_as)

plt.show()
    
    
    
    