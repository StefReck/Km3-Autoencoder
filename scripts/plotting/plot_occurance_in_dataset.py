# -*- coding: utf-8 -*-

"""
Plot number of events vs Energy, binned.
and # up events vs Energy, binned.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

import sys
#sys.path.append('scripts/util/')
sys.path.append('../util/')
from saved_setups_for_plot_statistics import get_plot_statistics_plot_size

datafile="data/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
save_to="results/plots/stats_train_muon-CC_and_elec-CC_each_240_xzt_shuffled.pdf"
save_to_down="results/plots/stats_train_muon-CC_and_elec-CC_each_240_xzt_shuffled_down.pdf"

debug=1


if debug:
    energies=np.random.rand(10000)*100
    downgoing=np.random.randint(0,2,size=10000)
else:
    file=h5py.File(datafile, "r")
    mc_info=file["y"]
    #event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, 
    #dir_x/y/z -> 5/6/7, time -> 8]
    energies=mc_info[:,2]
    downgoing = mc_info[:,7]<0
    
def make_plot(energies):
    ylog=True
    xlog=False
    
    figsize, fontsize = get_plot_statistics_plot_size("two_in_one_line")
    plt.rcParams.update({'font.size': fontsize})
    
    fig, ax = plt.subplots(figsize=figsize)
    if ylog: 
        plt.yscale("log")
    if xlog:  
        plt.xscale("log")
        energy_bins=np.logspace(np.log10(3),np.log10(100),98)
    else:
        energy_bins=np.linspace(3,100,98)
    ax.hist(energies, energy_bins, histtype="step", lw=2)
    plt.grid()
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Number of Events")
    plt.subplots_adjust(left=0.2)
    return fig


def make_plot_down(downgoing):
    figsize, fontsize = get_plot_statistics_plot_size("two_in_one_line")
    plt.rcParams.update({'font.size': fontsize})
    
    fig, ax = plt.subplots(figsize=figsize)
    energy_bins=np.linspace(3,100,98)
    
    ax.hist(downgoing, energy_bins, histtype="step", lw=2, density=True)
    plt.grid()
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Fraction of down-going events")
    plt.subplots_adjust(left=0.2)
    return fig


#fig = make_plot(energies)
fig_down = make_plot_down(downgoing)

plt.show(fig_down)

fig.savefig(save_to)
print("Plot saved to", save_to)
fig_down.save_fig(save_to_down)
print("Plot saved to", save_to_down)





