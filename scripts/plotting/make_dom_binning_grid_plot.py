# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mpl_toolkits.mplot3d import Axes3D

z_layer_no=5 #number of z layer to plot xy from, bottom to top
geo_file = "../Daten/ORCA_Geo_115lines.txt"
n_bins=(11,13,18)

x_bin_offset=0 #-2,0 ist gut!
y_bin_offset=0
#[id,x,y,z]
geo = np.loadtxt(geo_file)


x=geo[:,1]
y=geo[:,2]
z=geo[:,3]

which_z_layer = z==np.unique(z)[z_layer_no]
geo_reduced = geo[which_z_layer,:]
x_red=geo_reduced[:,1]
y_red=geo_reduced[:,2]

def calculate_bin_edges(n_bins, geo):
    """
    Calculates the bin edges for the later np.histogramdd actions based on the number of specified bins. 
    This is performed in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.
    :param tuple n_bins: contains the desired number of bins for each dimension. [n_bins_x, n_bins_y, n_bins_z]
    :param str fname_geo_limits: filepath of the .txt ORCA geometry file.
    :return: ndarray(ndim=1) x_bin_edges, y_bin_edges, z_bin_edges: contains the resulting bin edges for each dimension.
    """
    #print "Reading detector geometry in order to calculate the detector dimensions from file " + fname_geo_limits
    #geo = np.loadtxt(fname_geo_limits)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis = 0), np.nanmax(geo, axis = 0)
    #print ('Detector dimensions [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]: ' + str(geo_limits))

    x_bin_edges = np.linspace(geo_limits[0][1] - 9.95, geo_limits[1][1] + 9.95, num=n_bins[0] + 1) #try to get the lines in the bin center 9.95*2 = average x-separation of two lines
    y_bin_edges = np.linspace(geo_limits[0][2] - 9.75, geo_limits[1][2] + 9.75, num=n_bins[1] + 1) # Delta y = 19.483
    z_bin_edges = np.linspace(geo_limits[0][3] - 4.665, geo_limits[1][3] + 4.665, num=n_bins[2] + 1) # Delta z = 9.329

    #calculate_bin_edges_test(geo, y_bin_edges, z_bin_edges) # test disabled by default. Activate it, if you change the offsets in x/y/z-bin-edges

    return x_bin_edges, y_bin_edges, z_bin_edges

x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges(n_bins, geo)
x_bin_edges+=x_bin_offset
y_bin_edges+=y_bin_offset

def calculate_bin_stats(x_one_layer, y_one_layer, x_bin_edges, y_bin_edges):
    hist_xy = np.histogram2d(x_one_layer, y_one_layer, bins=(x_bin_edges, y_bin_edges))[0]
    unique, counts = np.unique(hist_xy.flatten(), return_counts=True)
    return unique,counts

def scan_for_optimum(x_one_layer, y_one_layer, n_bins, geo):
    x_edges, y_edges, z_edges = calculate_bin_edges(n_bins, geo)
    current_unique, current_counts=calculate_bin_stats(x_one_layer, y_one_layer, x_edges, y_edges)
    
    result_matrix=np.zeros((3,10,10), dtype=int)
    
    for x_off in np.arange(-5,5,1):
        for y_off in np.arange(-5,5,1):
            unique,counts =calculate_bin_stats(x_one_layer, y_one_layer, x_edges+x_off, y_edges+y_off)
            result_matrix[0,x_off+5,y_off+5]=counts[0]
            result_matrix[1,x_off+5,y_off+5]=counts[1]
            result_matrix[2,x_off+5,y_off+5]=counts[2] if len(counts)>=3 else 0
            print(x_off,y_off,"\t", counts)
    for i in range(len(result_matrix[0])):
        for j in range(len(result_matrix[0,0])):
            print (result_matrix[:,i,j], end="")
        print("")

#scan_for_optimum(x_red,y_red, n_bins, geo)

def plot_3d(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_2d(x_bin_edges, y_bin_edges, x_one_layer, y_one_layer):
    #xy_one_layer: x and y coordinates of doms of one z layer
    box_length_x=x_bin_edges[1]-x_bin_edges[0]
    box_length_y=y_bin_edges[1]-y_bin_edges[0]
    hist_xy = np.histogram2d(x_one_layer, y_one_layer, bins=(x_bin_edges, y_bin_edges))[0]
    print(calculate_bin_stats(x_one_layer, y_one_layer, x_bin_edges, y_bin_edges))
    
    #max_doms_inside=hist_xy.max() #2
    #min_doms_inside=hist_xy.min() #0
    
    fig = plt.figure(figsize=(8,13))
    ax = fig.add_subplot(111)
    
    for x_bin_edge in x_bin_edges:
        ax.plot([x_bin_edge,x_bin_edge], [y_bin_edges.min(),y_bin_edges.max()], color="black", ls="-", zorder=-1) 
    for y_bin_edge in y_bin_edges:
        ax.plot([x_bin_edges.min(),x_bin_edges.max()], [y_bin_edge,y_bin_edge], color="black", ls="-", zorder=-1) 
    
    for bin_no_x, x_bin_edge in enumerate(x_bin_edges[:-1]):
        for bin_no_y, y_bin_edge in enumerate(y_bin_edges[:-1]):
            alpha_max=0.3
            doms_inside = hist_xy[bin_no_x,bin_no_y]
            alpha = doms_inside * alpha_max / 2
            #alpha = (doms_inside-min_doms_inside) * alpha_max/max_doms_inside
            ax.add_patch(Rectangle([x_bin_edge, y_bin_edge], box_length_x, box_length_y, fc="blue", alpha=alpha, zorder=-2))
            
            
    plt.rcParams.update({'font.size': 16})
    ax.scatter(x_one_layer, y_one_layer, c='r', marker='o', label="DOM lines", zorder=1)
    ax.set_xlabel('X (m)')
    ax.minorticks_on()
    ax.set_ylabel('Y (m)')
    ax.set_aspect("equal")
    
    new_tick_locations_x = x_bin_edges[:-1] + 0.5*box_length_x
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations_x)
    ax2.set_xticklabels(np.arange(1,n_bins[0]+1,1))
    ax2.set_xlabel("x bin no.")
    ax2.set_aspect("equal")
    
    new_tick_locations_y = y_bin_edges[:-1] + 0.5*box_length_y
    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.set_yticks(new_tick_locations_y)
    ax3.set_yticklabels(np.arange(1,n_bins[1]+1,1))
    ax3.set_ylabel("y bin no.")
    ax3.set_aspect("equal")
    
    legend = ax.legend(loc="lower right")
    legend.get_frame().set_alpha(1)
    fig.suptitle("Binning and DOM locations")
    plt.show()
    
plot_2d(x_bin_edges, y_bin_edges, x_red, y_red)

