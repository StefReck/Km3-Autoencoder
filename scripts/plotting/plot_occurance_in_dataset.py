# -*- coding: utf-8 -*-

"""
Plot number of events vs Energy, binned.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Plot number of events vs Energy, binned.')
    parser.add_argument('datafile', nargs="+", type=str, help='Datafile')

    args = parser.parse_args()
    return  args.datafile

debug=False

if not debug:
    datafile=parse_input()

ylog=True
xlog=True

if debug:
    energies=np.random.rand(10000)*100
else:
    file=h5py.File(datafile, "r")
    mc_info=file["y"]
    #event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, 
    #dir_x/y/z -> 5/6/7, time -> 8]
    energies=mc_info[:,2]
    
if ylog: 
    plt.yscale("log")
if xlog:  
    plt.xscale("log")
    energy_bins=np.logspace(np.log10(3),np.log10(100),98)
else:
    energy_bins=np.linspace(3,100,98)
plt.hist(energies, energy_bins)





