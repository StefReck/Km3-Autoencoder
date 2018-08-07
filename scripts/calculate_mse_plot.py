# -*- coding: utf-8 -*-
"""
Plot logfile made from calculate mse.
"""
import argparse

def unpack_parsed_args():
    parser = argparse.ArgumentParser(description='Plot logfile made from calculate mse.')
    parser.add_argument('logfile_path', type=str, help='Path to the logfile created by calculate_mse, or "auto" to make and save all plots.')
    
    args = parser.parse_args()
    params = vars(args)
    return params["logfile_path"]

file=unpack_parsed_args()

import matplotlib.pyplot as plt

from plotting.plot_statistics import read_out_file, get_proper_range


def make_plot(data_dict, figsize=(6.4,5.5), title="Constituents of the test MSE"):
    plt.rcParams.update({'font.size': 14})
    #whats_there = ("MSE above3","MSE below","MSE")
    labels = ("High count", "Low count", "Total")
    colors = ("green","orange","blue")
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.title("")
    plt.grid()
    marksize=3
    ax.plot(data_dict["Epoch"], data_dict["MSE below"], "o-", label=labels[1], c=colors[1], ms=marksize)
    ax.plot(data_dict["Epoch"], data_dict["MSE"], "o-", label=labels[2], c=colors[2], ms=marksize)
    ax.set_xlabel("Autoencoder epoch")
    ax.set_ylabel("Test MSE (low counts and total)")
    plt.legend(loc="upper left")
    
    
    ax2 = ax.twinx()
    ax2.plot(data_dict["Epoch"], data_dict["MSE above3"], "o-", label=labels[0], c=colors[0], ms=marksize)
    ax2.set_ylabel("Test MSE (high counts)")
    
    ax.set_ylim(get_proper_range( list(map(float, data_dict["MSE"]))))
    ax2.set_ylim([0,11])
    
    plt.legend(loc="upper right")
    plt.subplots_adjust(left=0.15)
    return fig

def get_data_infos():
    loss_analysis=["mse_trained_vgg_5_picture-instanthighlr_autoencoders.txt",
                   "mse_trained_vgg_5_picture-instanthighlr_msep_autoencoders.txt",
                   "mse_trained_vgg_5_picture-instanthighlr_msep2_autoencoders.txt",
                   "mse_trained_vgg_5_picture-instanthighlr_msepsq_autoencoders.txt"]
    titles=["Constituents of the test MSE",
            "Constituents of the test MSE (MSEp training)",
            "Constituents of the test MSE (MSEp training, low lr)",
            r"Constituents of the test MSE (MSEp$^2$ training)"]
    
    
    data_path="/home/woody/capn/mppi013h/Km3-Autoencoder/results/"
    filenames = [data_path+"plots/mse/"+l[:-4]+".pdf" for l in loss_analysis]
    files = [data_path+"data/"+l for l in loss_analysis]
    
    return files, titles, filenames


if file=="auto":
    files, titles, filenames = get_data_infos()
    print("Generating plots automatically.")
    for i,f in enumerate(files):
        data_dict=read_out_file(f)
        fig = make_plot(data_dict, title=titles[i])
        plt.savefig(filenames[i])
        print("Saved plot as", filenames[i])
    print("Done.")
else:
    data_dict=read_out_file(file)
    fig = make_plot(data_dict)
    plt.show(fig)


