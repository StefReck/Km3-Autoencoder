# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def drawSphere(xCenter, yCenter, zCenter, r, col='r'):
    #draw sphere
    #u, v = np.mgrid[0:2*np.pi:2j, 0:np.pi:10j]
    u = np.linspace(0, 2*np.pi, 6)
    v = np.linspace(0, np.pi, 3)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    ax.plot_surface(x, y, z, color=col)
    #return (x,y,z)

#Bins: 
#n_bins=[11,13,18,50]


def reshape_4d_to_3d(hist_data):
    #input: 11x13x18x50
    #output: [ [x],[y],[z],[val] ]
    n_bins = hist_data.shape
    tot_bin_no=1
    for i in range(len(n_bins)-1):
        tot_bin_no=tot_bin_no*n_bins[i]
        
    grid=np.zeros((4,tot_bin_no)).astype('int')
    i=0
    for x in range( n_bins[0] ):
        for y in range( n_bins[1] ):
            for z in range( n_bins[2] ):
                val=0
                for t in range( n_bins[3] ):
                    val=val+hist_data[x][y][z][t]
                grid[0][i]=x
                grid[1][i]=y
                grid[2][i]=z
                grid[3][i]=val
                i=i+1
    return grid
           
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
                     
def reshape_2d_to_2d(hist_data):
    #input: a x b
    #output: [ [a],[b],[val] ]
    n_bins=hist_data.shape
    tot_bin_no=1
    for i in range(len(n_bins)):
        tot_bin_no=tot_bin_no*n_bins[i]
        
    grid=np.zeros((3,tot_bin_no)).astype('int')
    i=0
    for x in range( n_bins[0] ):
        for y in range( n_bins[1] ):
            val=hist_data[x][y]
            grid[0][i]=x
            grid[1][i]=y
            grid[2][i]=val
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

def make_2d_plot(grid, title):
    #input format: [a,b,val] 3 x bins
    #maximale count zahl pro bin
    max_value=max(l for l in grid.flatten())
    fraction=grid[-1]/max_value

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot = ax.scatter(grid[0],grid[1], c=grid[-1], s=300*fraction)
    cbar=fig.colorbar(plot)
    cbar.set_label('Hits', rotation=270)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    plt.show()



#shape: 1,11,13,18,50
all_4d_to_4d_hists = np.load("Daten/test_hist_4d.npy")

#shape: 1,4,11,13,18
all_4d_to_3d_hists = np.load("Daten/test_hist_3d.npy")

all_4d_to_2d_hists = np.load("Daten/test_hist_2d.npy")



#2D Histograms:
xy_hist = all_4d_to_2d_hists[0][0]
xz_hist = all_4d_to_2d_hists[0][1]
yz_hist = all_4d_to_2d_hists[0][2]
#...

#3D histograms:
"""
file=h5py.File('Daten/xyz.h5', 'r')
xyz_hist = np.array(file["x"])[0]
"""
xyz_hist = all_4d_to_3d_hists[0][0]
xyt_hist = all_4d_to_3d_hists[0][1]
xzt_hist = all_4d_to_3d_hists[0][2]
yzt_hist = all_4d_to_3d_hists[0][3]

#4D histogram:
#11,13,18,50
xyzt_hist = all_4d_to_4d_hists[0]

def show_hist_info():
    print("Total Hits xyz:", xyz_hist.flatten().sum())
    print("Total Hits xyt:", xyt_hist.flatten().sum())
    print("Total Hits xzt:", xzt_hist.flatten().sum())
    print("Total Hits yzt:", yzt_hist.flatten().sum())
    print("Sum of above 4:", xyz_hist.flatten().sum() + xyt_hist.flatten().sum() + xzt_hist.flatten().sum() + yzt_hist.flatten().sum())

    print("Total Hits xyzt:", xyzt_hist.flatten().sum())

def sanity_check_4d(bin_number):
    print("Coordinates of bin %d in the reshaped 3d-from-4d histogram:" % bin_number)
    co = np.zeros(4).astype('int')
    for i in range(4):
        co[i]=grid_3d_from_4d[i][bin_number]
    print(co)
    print("The original 4d histogramm at these coordinates is:")
    print(xyzt_hist[ co[0] ][ co[1] ][ co[2] ])
    print("The sum of which is:")
    print(xyzt_hist[ co[0] ][ co[1] ][ co[2] ].sum())
    print("The 3d xyz-histogram, however, is at these coordinates:")
    print(xyz_hist[ co[0] ][ co[1] ][ co[2] ])
    print("The reshaped xyz Histogram is:")
    print(grid_3d_from_xyz[3][bin_number])



#show_hist_info()
grid_3d_from_4d=reshape_4d_to_3d(xyzt_hist)
grid_3d_from_xyz=reshape_3d_to_3d(xyz_hist)
grid_2d_from_xy=reshape_2d_to_2d(xy_hist)

sanity_check_4d(1824)


make_3d_plot(grid_3d_from_xyz, "From 3D Hist")
#make_3d_plot(grid_3d_from_4d, "From 4d Hist")
#make_2d_plot(grid_2d_from_xy, "XY")


