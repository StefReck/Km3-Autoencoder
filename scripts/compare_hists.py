# -*- coding: utf-8 -*-
"""
Make 3D Scatter Histogram from 11x13x18 np array, and compare to another one.
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

def make_3d_plot(grid, title):
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
    ax.set_title(title)

    plt.show()

def compare_hists(hist_org, hist_pred):
    
    make_3d_plot(reshape_3d_to_3d(hist_org), "Original")
    make_3d_plot(reshape_3d_to_3d(hist_pred), "Prediction")

