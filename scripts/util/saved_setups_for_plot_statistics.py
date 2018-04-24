# -*- coding: utf-8 -*-
"""
Some saved setups for the plot ststistics scripts.
"""
import numpy as np

def get_list_of_all_parallel_tags():
    the_list = ["msep","msepsq","msep2",
                "channel_3n_noZeroEvent","channel_3n","channel_3n_noZero",
                "channel_5n", "channel_10n", 
                "vgg_3", "vgg_5_600_picture", "vgg_5_600_morefilter",
                "vgg_5_600_channel", "vgg_5_200", "vgg_5_200_dense", "vgg_5_64", "vgg_5_32", 
                "vgg_5_32-eps01", 
                "vgg_5_200_deep", "vgg_5_200_large", "vgg_5_200_shallow", "vgg_5_200_small", 
                "vgg_3_energy", "vgg_5_200_energy", "vgg_5_64_energy", "vgg_5_32-eps01_energy",]
    return the_list

def get_props_for_plot_parallel(tag):
    #For the script plots_statistics_parallel, which takes exactly two models
    #as an input (AE and parallel encoder)
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/"
    epoch_schedule="10-2-1"
    labels_override = ["Autoencoder", "Encoder"] 
    save_to_folder = ""
    #-------------------vgg5 picture loss functions tests---------------------------
    if tag=="msep":
        title = "Parallel training with MSEp autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "loss_functions/"
    elif tag=="msepsq":
        title = r"Parallel training with MSEp$^2$ autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "loss_functions/"   
    elif tag=="msep2":
        title = "Parallel training with MSEp autoencoder loss (low lr)"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "loss_functions/"
        
    #-------------------Channel Encoders---------------------------
    elif tag=="channel_3n_noZeroEvent":
        title = "Parallel training with channel autoencoder (3 neurons) and balanced dataset"
        ae_model =  home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"
    elif tag=="channel_3n":
        title = "Parallel training with channel autoencoder (3 neurons)"
        ae_model =  home+"models/channel_3n_m3/trained_channel_3n_m3_autoencoder_test.txt" 
        prl_model = home+"models/channel_3n_m3/trained_channel_3n_m3_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"    
    elif tag=="channel_3n_noZero":
        title = "Parallel training with channel autoencoder (3 neurons) and no zero centering"
        ae_model =  home+"models/channel_3n_m3-noZero/trained_channel_3n_m3-noZero_autoencoder_test.txt" 
        prl_model = home+"models/channel_3n_m3-noZero/trained_channel_3n_m3-noZero_autoencoder_supervised_parallel_up_down_dropout_stateful_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"  
        
    elif tag=="channel_5n":
        title = "Parallel training with channel autoencoder (5 neurons)"
        ae_model =  home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"
    elif tag=="channel_10n":
        title = "Parallel training with channel autoencoder (10 neurons)"
        ae_model =  home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/" 
        epoch_schedule="1-1-1"
        
    #-------------------Up Down classifications vgg_5---------------------------
    elif tag=="vgg_3":
        title = "Parallel training with model '2000'"
        ae_model =  home+"models/vgg_3/trained_vgg_3_autoencoder_test.txt" 
        prl_model = home+"models/vgg_3/trained_vgg_3_autoencoder_supervised_parallel_up_down_test.txt"
        epoch_schedule="10-1-1"
        save_to_folder = "bottleneck/"
    
    elif tag=="vgg_5_600_picture":
        title = "Parallel training with model '600 picture'"
        ae_model =  home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_up_down_new_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_600_morefilter":
        title = "Parallel training with model '600 morefilter'"
        ae_model =  home+"models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_600_channel":
        title = "Parallel training with model '600 channel'"
        ae_model =  home+"models/vgg_5_channel/trained_vgg_5_channel_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_channel/trained_vgg_5_channel_autoencoder_supervised_parallel_up_down_dropout06_test.txt"
        print("60 % Dropout was used for this model. Others are available, but the overfitting is best seen here.")
        save_to_folder = "bottleneck/"
        
    elif tag=="vgg_5_200":
        title = "Parallel training with model '200'"
        ae_model =  home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_200_dense":
        title = "Parallel training with model '200 dense'"
        ae_model =  home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"      
        
    elif tag=="vgg_5_64":
        title = "Parallel training with model '64'"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
        
    elif tag=="vgg_5_32":
        title = "Parallel training with model '32' and $\epsilon=10^{-8}$"
        ae_model =  home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_32-eps01":
        title = "Parallel training with model '32'"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
        
    #-------------------vgg_5_200 Parameter Erhöhung---------------------------
    elif tag=="vgg_5_200_deep":
        title = "Parallel training with model '200 deep'"
        ae_model =  home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
    elif tag=="vgg_5_200_large":
        title = "Parallel training with model '200 large'"
        ae_model =  home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
    elif tag=="vgg_5_200_shallow":
        title = "Parallel training with model '200 shallow'"
        ae_model =  home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
    elif tag=="vgg_5_200_small":
        title = "Parallel training with model '200 small'"
        ae_model =  home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
        
    #-------------------Energy reconstructions---------------------------
    elif tag=="vgg_3_energy":
        title = "Parallel training with model '2000'"
        ae_model =  home+"models/vgg_3/trained_vgg_3_autoencoder_test.txt" 
        prl_model = home+"models/vgg_3/trained_vgg_3_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_600_picture_energy":
        title = "Parallel training with model '600 picture'"
        ae_model =  home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_200_energy":
        title = "Parallel training with model '200'"
        ae_model =  home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_energy_linear_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_64_energy":
        title = "Parallel training with model '64'"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_32-eps01_energy":
        title = "Parallel training with model '32'"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
        
    else:
        raise NameError("Tag "+tag+" unknown.")
        
    test_files=[ae_model, prl_model]
    print("Loaded the following files:")
    for file in test_files:
        print(file.split(home)[1]) 
        
    save_as=home+"results/plots/statistics/"+save_to_folder+"statistics_parallel_"+prl_model.split("/")[-1][:-4]+".pdf"
    return test_files, title, labels_override, save_as, epoch_schedule


def get_how_many_epochs_each_to_train(epoch_schedule):
    #Which epochs from the parallel encoder history to take:
    if epoch_schedule=="1-1-1":
        how_many_epochs_each_to_train = np.ones(100).astype(int)
    elif epoch_schedule=="10-2-1":
        how_many_epochs_each_to_train = np.array([10,]*1+[2,]*5+[1,]*200)
    elif epoch_schedule=="10-1-1":
        #Was used once for vgg_3_parallel
        how_many_epochs_each_to_train = np.array([10,]*5+[1,]*200)
    print("Using parallel schedule", how_many_epochs_each_to_train[:12,], "...")
    return how_many_epochs_each_to_train


def get_props_for_plot_parser(tag):
    #For the script plots_statistics_parser
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/"
    legend_locations=(1, "upper left")
    
    if tag=="channel-encs":
        title = "Encoder performance of channel autoencoder networks"
        test_files=[home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_epoch35_supervised_up_down_stateful_convdrop_test.txt", 
                    home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_epoch17_supervised_up_down_stateful_convdrop_test.txt",
                    home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_epoch23_supervised_up_down_stateful_convdrop_test.txt"]
        labels_override = ["3 neurons", "5 neurons", "10 neurons"]
        legend_locations=("lower right", "upper left")
    
    else:
        raise NameError("Tag "+tag+" unknown.")
        
    print("Loaded the following files:")
    for file in test_files:
        print(file.split(home)[1]) 
        
    save_as=home+"results/plots/statistics/statistics_parser_"+tag+".pdf"
    return test_files, title, labels_override, save_as, legend_locations
