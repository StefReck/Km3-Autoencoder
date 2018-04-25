# -*- coding: utf-8 -*-
"""
This shows the impact of broken_data_mode in 3d plots of events.
They are taken from the generator (on the xzt_debug dataset), just like in training.
This script is meant to quickly plot the ones that dont require their own dataset (aka can be manipulated on the fly)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from util.run_cnn import generate_batches_from_hdf5_file
from get_dataset_info import get_dataset_info
from plotting.histogramm_3d_utils import make_3d_plots, reshape_3d_to_3d, get_title_arrays

which_broken_mode=12
#How many to display
batchsize=3
#save to data; None for dont save but just display
save_to_pdf_name=None#"broken_2_3D_hist.pdf"
#norm = normal first for plot, inv = broken first
order="norm"
#Redraw events from the file until at least one has an energy higher then this
minimum_energy = 75

#Minimum counts for the plots
min_counts=0.1
#File that supplies the data
dataset_tag="debug_xzt"




if which_broken_mode==1:
    suptitle_base="Simulated data with added up-down information"
elif which_broken_mode==2:
    suptitle_base="Simulated data with added noise"
    #cbar gets weird here, maybe duplicate cbar1?
elif which_broken_mode==4:
    suptitle_base="Simulated data with more counts for up-going events"
elif which_broken_mode==12:
    suptitle_base="Simulated data with added energy-correlated noise"
    
if order=="inv":
    titles=["Manipulated simulation", "Original simulation as 'measured' data"]
elif order=="norm":
    titles=["Original simulation", "Manipulated simulation as 'measured' data"]

dataset_info_dict=get_dataset_info(dataset_tag)
is_autoencoder=True
class_type=None
yield_mc_info = True

filepath=dataset_info_dict["train_file"]
zero_center_image=dataset_info_dict["zero_center_image"]
n_bins = dataset_info_dict["n_bins"]
xs_mean=np.load(zero_center_image)

#Generator for Normal Data
generator_normal = generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, dataset_info_dict,
                                            broken_simulations_mode=0, f_size=None, zero_center_image=xs_mean,
                                            yield_mc_info=yield_mc_info, swap_col=None, is_in_test_mode = False)

#get data
while True:
    xs, mc_info = next(generator_normal)[1:]
    # content of mc_info:
    # [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]
    energies = mc_info[:,2]
    if (energies>minimum_energy).sum() >= 1:
        break

#Generator for Broken Data
dataset_info_dict["broken_simulations_mode"]=which_broken_mode
generator_broken = generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, dataset_info_dict,
                                            broken_simulations_mode=which_broken_mode, f_size=None, zero_center_image=xs_mean,
                                            yield_mc_info=True, swap_col=None, is_in_test_mode = False)

#get data
while True:
    xs_broken, mc_info = next(generator_broken)[1:]
    # content of mc_info:
    # [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]
    energies = mc_info[:,2]
    if (energies>minimum_energy).sum() >= 1:
        break
    

#add zero centering again
data_normal = np.add(xs, xs_mean)
if which_broken_mode==3:
    xs_mean[:,:6,:,:]=np.zeros_like(xs_mean[:,:6,:,:])
data_broken=np.add(xs_broken, xs_mean)

#infos for the title
title_array = get_title_arrays(mc_info)


#Make the plots
fig_array=[]
for i in range(len(data_normal)):
    plot_brok=reshape_3d_to_3d(data_broken[i], min_counts)
    plot_norm=reshape_3d_to_3d(data_normal[i], min_counts)
    suptitle = suptitle_base +"   ("+title_array[0][i]+", "+title_array[1][i]+", "+title_array[2][i]+")"
    if order=="norm":
        fig = make_3d_plots(plot_norm, n_bins[:-1], hist_pred=plot_brok, suptitle=suptitle, figsize=(12,7), titles=titles)
    elif order=="inv":
        fig = make_3d_plots(plot_brok, n_bins[:-1], hist_pred=plot_norm, suptitle=suptitle, figsize=(12,7), titles=titles)
    else:
        raise()
    fig_array.append(fig)
    #plt.close(fig)


#Show and save the plots
if save_to_pdf_name == None:
    for figure in fig_array:
        plt.show(figure)
else:
    with PdfPages(save_to_pdf_name) as pp:
        for figure in fig_array:
            pp.savefig(figure)
            plt.close(figure)
    print("Saved as", save_to_pdf_name)

        
    

