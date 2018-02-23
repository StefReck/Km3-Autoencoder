# -*- coding: utf-8 -*-
"""
This shows the impact of broken_data_mode in 3d plots of events.
They are taken from the generator, just like in training.
"""
import h5py
import numpy as np

from util.run_cnn import generate_batches_from_hdf5_file
from get_dataset_info import get_dataset_info
from plotting.make_autoencoder_3d_output_plot import make_3d_plots, reshape_3d_to_3d
import matplotlib.pyplot as plt

which_broken_mode=4
batchsize=5
min_counts=0.1

titles=["Manipulated simulation", "Original simulation as 'measured' data"]
suptitle="Simulated data with added up-down information"

#norm = normal first for plot, inv = broken first
order="norm"

dataset_info_dict=get_dataset_info("debug_xzt")
is_autoencoder=True
class_type=None

yield_mc_info = True



filepath=dataset_info_dict["train_file"]
zero_center_image=dataset_info_dict["zero_center_image"]
n_bins = dataset_info_dict["n_bins"]

xs_mean=np.load(zero_center_image)

generator_normal = generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, dataset_info_dict,
                                            broken_simulations_mode=0, f_size=None, zero_center_image=xs_mean,
                                            yield_mc_info=yield_mc_info, swap_col=None, is_in_test_mode = False)
xs, xs2, mc_info = next(generator_normal)
data_normal = np.add(xs, xs_mean)
# content of ys:
# [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]
print(mc_info)

dataset_info_dict["broken_simulations_mode"]=which_broken_mode

generator_broken = generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, dataset_info_dict,
                                            broken_simulations_mode=which_broken_mode, f_size=None, zero_center_image=xs_mean,
                                            yield_mc_info=False, swap_col=None, is_in_test_mode = False)
if which_broken_mode==3:
    xs_mean[:,:6,:,:]=np.zeros_like(xs_mean[:,:6,:,:])
data_broken=np.add(next(generator_broken)[0], xs_mean)

for i in range(len(data_normal)):
    plot_brok=reshape_3d_to_3d(data_broken[i], min_counts)
    plot_norm=reshape_3d_to_3d(data_normal[i], min_counts)
    if order=="norm":
        fig = make_3d_plots(plot_norm, plot_brok, n_bins[:-1], suptitle=suptitle, figsize=(12,7), titles=titles)
    elif order=="inv":
        fig = make_3d_plots(plot_brok, plot_norm, n_bins[:-1], suptitle=suptitle, figsize=(12,7), titles=titles)
    else:
        raise()
plt.show()

