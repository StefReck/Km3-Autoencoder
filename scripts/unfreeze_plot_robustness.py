# -*- coding: utf-8 -*-
"""
Makes the plot for the unfreeze test.
"""
import matplotlib.pyplot as plt

from plotting.plot_statistics import read_out_file

encoder_file="models/vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_test.txt"
#logfile_path="models/vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_unfreeze_broken4_log.txt"
logfile_path="trained_vgg_5_200-unfreeze_autoencoder_epoch1_unfreeze_broken4_log.txt"

data = read_out_file(logfile_path)
#This is present in the datafile, in addition to "epoch"
keys_robust = ['(Sim-Meas)', '(Upperlim-']
labels_robust=["Decrease simulations", "Decrease real"]
colors_robust=["yellowgreen", "darkolivegreen"]

keys_acc = ['acc_sim', 'acc_meas', 'acc_ulim' ]
labels_acc=["On 'simulations'", "On 'measured' data", "Upper limit on 'measured' data"]
colors_acc=["orange", "blue", "navy"]

fig, ax = plt.subplots(figsize=(9,6))
ax2 = ax.twinx()

fig.suptitle("Partially unfrozen model")
ax.set_xlabel("Epoch")
ax.xaxis.grid()
ax.set_ylabel("Accuracy")
ax2.set_ylabel("Percent")

for no,key in enumerate(keys_acc):
    ax.plot(data["epoch"], data[key], "o-", label=labels_acc[no], color=colors_acc[no])
ax.legend(loc="lower left")
for no,key in enumerate(keys_robust):
    ax2.plot(data["epoch"], data[key], "o--", label=labels_robust[no], color=colors_robust[no])
ax2.legend(loc="lower right")

plt.show()
    
    
    
    