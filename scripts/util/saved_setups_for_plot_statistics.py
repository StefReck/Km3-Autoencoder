# -*- coding: utf-8 -*-
"""
Some saved setups for the plot ststistics scripts.
"""


def get_props_for_plot_parallel(tag):
    #For the script plots_statistics_parallel, which takes exactly two models
    #as an input (AE and parallel encoder)
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/"
    epoch_schedule="10-2-1"
    if tag=="msep":
        title = "Parallel training with MSEp autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]
    elif tag=="msepsq":
        title = r"Parallel training with MSEp$^2$ autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]    
    elif tag=="msep2":
        title = "Parallel training with MSEp autoencoder loss (low lr)"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        
    elif tag=="channel_3n_noZeroEvent":
        title = "Parallel training with channel autoencoder (3 neurons) and balanced dataset"
        ae_model =  home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        epoch_schedule="1-1-1"
    elif tag=="channel_3n":
        title = "Parallel training with channel autoencoder (3 neurons)"
        ae_model =  home+"models/channel_3n_m3/trained_channel_3n_m3_autoencoder_test.txt" 
        prl_model = home+"models/channel_3n_m3/trained_channel_3n_m3_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        epoch_schedule="1-1-1"    
    elif tag=="channel_3n_noZero":
        title = "Parallel training with channel autoencoder (3 neurons) and no zero centering"
        ae_model =  home+"models/channel_3n_m3-noZero/trained_channel_3n_m3-noZero_autoencoder_test.txt" 
        prl_model = home+"models/channel_3n_m3-noZero/trained_channel_3n_m3-noZero_autoencoder_supervised_parallel_up_down_dropout_stateful_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        epoch_schedule="1-1-1"  
        
    elif tag=="channel_5n":
        title = "Parallel training with channel autoencoder (5 neurons)"
        ae_model =  home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        labels_override = ["Autoencoder", "Encoder"]  
        epoch_schedule="1-1-1"
    elif tag=="channel_10n":
        title = "Parallel training with channel autoencoder (10 neurons)"
        ae_model =  home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        labels_override = ["Autoencoder", "Encoder"]  
        epoch_schedule="1-1-1"
        
    elif tag=="vgg_5_200":
        title = "Parallel training with model '200'"
        ae_model =  home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]  
    elif tag=="vgg_5_200_dense":
        title = "Parallel training with model '200 dense'"
        ae_model =  home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"]       
        
    elif tag=="vgg_5_200_deep":
        title = "Parallel training with model '200 deep'"
        ae_model =  home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_200_large":
        title = "Parallel training with model '200 large'"
        ae_model =  home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_200_shallow":
        title = "Parallel training with model '200 shallow'"
        ae_model =  home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_200_small":
        title = "Parallel training with model '200 small'"
        ae_model =  home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        
         
    elif tag=="vgg_5_64":
        title = "Parallel training with model '64'"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_32":
        title = "Parallel training with model '32' and $\epsilon=10^{-8}$"
        ae_model =  home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
    elif tag=="vgg_5_32-eps01":
        title = "Parallel training with model '32'"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_up_down_test.txt"
        labels_override = ["Autoencoder", "Encoder"] 
        
    else:
        print("Tag", tag, "unknown.")
        raise()
    test_files=[ae_model, prl_model]
    
    print("Loaded the following files:")
    for file in test_files:
        print(file.split(home)[1]) 
        
    save_as=home+"results/plots/statistics/statistics_parallel_"+prl_model.split("/")[-1][:-4]+".pdf"
    return test_files, title, labels_override, save_as, epoch_schedule



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
        print("Tag", tag, "unknown.")
        raise()
        
    print("Loaded the following files:")
    for file in test_files:
        print(file.split(home)[1]) 
        
    save_as=home+"results/plots/statistics/statistics_parser_"+tag+".pdf"
    return test_files, title, labels_override, save_as, legend_locations
