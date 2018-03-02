# -*- coding: utf-8 -*-
"""
Make a quick plot of the statistics of DOM hits for the channel ID AE.
"""

import numpy as np
import matplotlib.pyplot as plt

file = "elec-CC_and_muon-CC_xyzc_train_1_to_480_shuffled_0_statistics_fraction_0.02.npy"
max_hits = 6

array = np.load(file)
total_doms = np.sum(array)

for_plot = np.zeros(max_hits+1)
hits_array = np.zeros(max_hits+1)
for hits in range(max_hits):
    for_plot[hits]=100*array[hits]/total_doms
    hits_array[hits]=hits
    
for_plot[-1]=100*np.sum(array[max_hits:])/total_doms
hits_array[-1]=max_hits

xtick_labels = list(range(max_hits))+[str(max_hits)+" or more",]
xticks = np.arange(0,max_hits+1)

y_ticks = [0.1, 1, 10, 100]
ytick_labels = y_ticks

fig=plt.figure(figsize=(9,7))
plt.grid(zorder=-10)

#plt.plot(hits_array, for_plot, "o-")
#plt.semilogy(hits_array, for_plot, "o-")
plt.bar(hits_array, for_plot, zorder=10)
plt.yscale("log")

plt.xticks(xticks, xtick_labels)
plt.yticks(y_ticks, ytick_labels)
plt.suptitle("Total number of hits per DOM")
plt.xlabel("Hits per DOM")
plt.ylabel("Percentage of total DOMs")