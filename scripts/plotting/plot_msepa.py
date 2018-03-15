# -*- coding: utf-8 -*-
"""
Show what the poisson mse does.
"""
import numpy as np
from scipy.special import factorial
import h5py
import matplotlib.pyplot as plt

from histogramm_3d_utils import get_title_arrays

def poisson_factor(y_true):
    expec = np.mean( y_true )
    poisson_factor = np.exp(-1 * expec)*np.power(expec,y_true)/factorial(y_true)
    return 1-poisson_factor

def poisson_square_factor(y_true):
    expec = np.mean( y_true )
    poisson_factor = np.exp(-1 * expec)*np.power(expec,y_true)/factorial(y_true)
    return np.square(1-poisson_factor)

def poisson_log_factor(y_true):
    expec = np.mean( y_true )
    poisson_factor = np.exp(-1 * expec)*np.power(expec,y_true)/factorial(y_true)
    return -(1/100)*np.log(poisson_factor)

home_path="../"
data_path=home_path+"Daten/xzt/"
train_data="JTE_KM3Sim_gseagen_elec-CC_3-100GeV-1_1E6-1bin-3_0gspec_ORCA115_9m_2016_100_xzt.h5"
file=data_path+train_data

test_file=h5py.File(file , 'r')

def plot_which(event_no, test_file):
    data = np.sum(test_file["x"][event_no], axis=0)
    info = test_file["y"][event_no: event_no+1]
    info_array = get_title_arrays(info)
    #(['Up-going'], ['Anti-Elec-CC'], ['Energy: 43.7 GeV'])
    #event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    poisson = poisson_factor(data)
    
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8,6))
    ax1.set_title("Original event:   " + info_array[0][0] + "   "+ info_array[1][0] +  "   "+info_array[2][0])
    ax1.set_ylabel("Z bin")
    ax1.set_xlabel("T bin")
    divide_by = np.max(data)
    org_plot = ax1.imshow(data/divide_by, )
    cbar = plt.colorbar(org_plot, ax=ax1)
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels(list(map(int,cbar.get_ticks()*divide_by)))
    cbar.ax.set_title('Hits')
    
    ax2.set_title("Poisson sensitivity")
    pois_plot = ax2.imshow(poisson)
    ax2.set_ylabel("Z bin")
    ax2.set_xlabel("T bin")
    
    cbar_2=plt.colorbar(pois_plot, ax=ax2)
    cbar_2.ax.set_title(r'$1 - P_{poisson}$')
    fig.tight_layout()
    return fig

def plot_which_quadro(event_no, test_file):
    data = np.sum(test_file["x"][event_no], axis=0)
    info = test_file["y"][event_no: event_no+1]
    info_array = get_title_arrays(info)
    #(['Up-going'], ['Anti-Elec-CC'], ['Energy: 43.7 GeV'])
    #event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(7,11))
    ax1.set_title("Original event:   " + info_array[0][0] + "   "+ info_array[1][0] +  "   "+info_array[2][0])
    ax1.set_ylabel("Z bin")
    ax1.set_xlabel("T bin")
    divide_by = np.max(data)
    org_plot = ax1.imshow(data/divide_by, )
    cbar = plt.colorbar(org_plot, ax=ax1)
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels(list(map(int,cbar.get_ticks()*divide_by)))
    cbar.ax.set_title('Hits')
    
    poisson = poisson_factor(data)
    ax2.set_title("Poisson sensitivity")
    pois_plot = ax2.imshow(poisson)
    ax2.set_ylabel("Z bin")
    ax2.set_xlabel("T bin")
    cbar_2=plt.colorbar(pois_plot, ax=ax2)
    cbar_2.ax.set_title(r'$1 - P_{poisson}$')
    
    poisson_sq = poisson_square_factor(data)
    ax3.set_title("Poisson squared sensitivity")
    pois_plot_3 = ax3.imshow(poisson_sq)
    ax3.set_ylabel("Z bin")
    ax3.set_xlabel("T bin")
    cbar_3=plt.colorbar(pois_plot_3, ax=ax3)
    cbar_3.ax.set_title(r'$(1 - P_{poisson})^{2}$')
    
    poisson_log = poisson_log_factor(data)
    ax4.set_title("Poisson log sensitivity")
    pois_plot_4 = ax4.imshow(poisson_log)
    ax4.set_ylabel("Z bin")
    ax4.set_xlabel("T bin")
    cbar_4=plt.colorbar(pois_plot_4, ax=ax4)
    cbar_4.ax.set_title(r'$-\log( P_{poisson}$)')
    
    fig.tight_layout()
    return fig



for i in [10,]:
    #fig = plot_which(i, test_file)
    fig = plot_which_quadro(i, test_file)
    plt.show(fig)

