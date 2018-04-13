# -*- coding: utf-8 -*-
"""
Look for events with the same event id in the datasets xzt and xzt_broken5 and plot them.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

from get_dataset_info import get_dataset_info
from plotting.histogramm_3d_utils import make_3d_plots, reshape_3d_to_3d, get_title_arrays
from plotting.histogramm_3d_utils import get_event_no_from_file, get_some_hists_from_file, make_plots_from_array

datainfo_xzt = get_dataset_info("xztc")
datainfo_xzt_broken5 = get_dataset_info("xzt_broken5")


train_file_xzt = datainfo_xzt["train_file"]
train_file_xzt_broken5 = datainfo_xzt_broken5["train_file"]

hists, labels = get_some_hists_from_file(train_file_xzt, 1, 50)
hists_b, labels_b = get_event_no_from_file(train_file_xzt_broken5, target_event_id=labels[0][0], event_track=None)

print(labels, labels_b)

#xztc hists haben shape (11,18,50,31)
hists=np.sum(hists[0], axis=-1)

fig = make_plots_from_array(hists, hists_b[0], suptitle="Broken 5", min_counts=0, titles=["Original","Manipulation"])
plt.show(fig)

