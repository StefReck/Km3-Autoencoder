# -*- coding: utf-8 -*-
"""
Make 3D Scatter Histogram from 11x13x18 np array, and compare to another one.
Can either plot single histogramms, or two side by side in one plot.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def reshape_3d_to_3d(hist_data):
    #input: 11x13x18
    #output: [ [x],[y],[z],[val] ]
    n_bins=hist_data.shape
    tot_bin_no=1
    for i in range(len(n_bins)):
        tot_bin_no=tot_bin_no*n_bins[i]
        
    grid=np.zeros((4,tot_bin_no)).astype('int')
    i=0
    for x in range( n_bins[0] ):
        for y in range( n_bins[1] ):
            for z in range( n_bins[2] ):
                val=hist_data[x][y][z]
                grid[0][i]=x
                grid[1][i]=y
                grid[2][i]=z
                grid[3][i]=val
                i=i+1
    return grid

def make_3d_plot(grid):
    #input format: [x,y,z,val]
    #maximale count zahl pro bin
    max_value=max(l for l in grid.flatten())
    fraction=grid[3]/max_value

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    #for (xi,yi,zi,val) in grid:
        #drawSphere(xi,yi,zi,r)

    plot = ax.scatter(grid[0],grid[1],grid[2], c=grid[3], s=8*36*fraction)
    cbar=fig.colorbar(plot)
    cbar.set_label('Hits', rotation=270)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def make_3d_plots(hist_org, hist_pred):
    #Plot original and predicted histogram side by side in one plot
    #input format: [x,y,z,val]
    #maximale count zahl pro bin


    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(121, projection='3d', aspect='equal')
    max_value1=max(l for l in hist_org.flatten())
    fraction1=hist_org[3]/max_value1
    plot1 = ax1.scatter(hist_org[0],hist_org[1],hist_org[2], c=hist_org[3], s=8*36*fraction1)
    cbar1=fig.colorbar(plot1,fraction=0.046, pad=0.1)
    cbar1.set_label('Hits', rotation=270, labelpad=0.1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title("Original")
    
    ax2 = fig.add_subplot(122, projection='3d', aspect='equal')
    max_value2=max(l for l in hist_pred.flatten())
    fraction2=hist_pred[3]/max_value2
    plot2 = ax2.scatter(hist_pred[0],hist_pred[1],hist_pred[2], c=hist_pred[3], s=8*36*fraction2)
    cbar2=fig.colorbar(plot2,fraction=0.046, pad=0.1)
    cbar2.set_label('Hits', rotation=270, labelpad=0.1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("Prediction")

    fig.tight_layout()
    plt.show()


def compare_hists(hist_org, hist_pred):
    make_3d_plots(reshape_3d_to_3d(hist_org), reshape_3d_to_3d(hist_pred))

def plot_hist(hist):
    make_3d_plot(reshape_3d_to_3d(hist))