# -*- coding: utf-8 -*-

"""
Plot number of events vs Energy, binned.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append('scripts/util/')
from saved_setups_for_plot_statistics import get_plot_statistics_plot_size

def parse_input():
    parser = argparse.ArgumentParser(description='Plot number of events vs Energy, binned.')
    parser.add_argument('datafile', type=str, help='Datafile')

    args = parser.parse_args()
    return  args.datafile

debug=0

if not debug:
    datafile=parse_input()


if debug:
    energies=np.random.rand(10000)*100
else:
    file=h5py.File(datafile, "r")
    mc_info=file["y"]
    #event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, 
    #dir_x/y/z -> 5/6/7, time -> 8]
    energies=mc_info[:,2]
    
def make_plot(energies):
    ylog=True
    xlog=1
    
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

fig = make_plot(energies)
plt.show(fig)



