# -*- coding: utf-8 -*-
"""
Some saved setups for the plot ststistics scripts.
"""
import numpy as np

def get_plot_statistics_plot_size(style):
    #Plot and font sizes for plot_statistics:
    if style=="two_in_one_line":
        #For putting 2 plots next to each other, 0.48\textwidth
        figsize = [6.4,5.5]   
        font_size=14
    elif style=="extended":
        #For a single plot in a line, 0.95\textwidth
        figsize = [10, 5.5] 
        font_size=12
    elif style=="double":
        #For two subplots horizontally next to each other, 0.95\textwidth
        figsize = [12.8, 5.5] 
        font_size=14

    return figsize, font_size

def get_how_many_epochs_each_to_train(epoch_schedule):
    #Which epochs from the parallel encoder history to take, depending on the 
    #string that get_props... returns
    if epoch_schedule=="1-1-1":
        how_many_epochs_each_to_train = np.ones(100).astype(int)
    elif epoch_schedule=="10-2-1":
        how_many_epochs_each_to_train = np.array([10,]*1+[2,]*5+[1,]*200)
    elif epoch_schedule=="10-1-1":
        #Was used once for vgg_3_parallel
        how_many_epochs_each_to_train = np.array([10,]*5+[1,]*200)
    print("Using parallel schedule", how_many_epochs_each_to_train[:12,], "...")
    return how_many_epochs_each_to_train


def get_props_for_plot_parallel(tag, printing=True):
    #For the script plots_statistics_parallel, which takes exactly two models
    #as an input (AE and parallel encoder)
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/"
    epoch_schedule="10-2-1"
    labels_override = ["Autoencoder", "Encoder"] 
    save_to_folder = ""
    #plot and font sizes:
    style="two_in_one_line"
    #plot title
    title=""
    
    try: tag=int(tag) 
    except: ValueError
    #-------------------vgg5 picture loss functions tests---------------------------
    if tag=="msep" or tag==0:
        title = "Parallel training with MSEp autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep/trained_vgg_5_picture-instanthighlr_msep_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "loss_functions/"
    elif tag=="msepsq" or tag==1:
        title = r"Parallel training with MSEp$^2$ autoencoder loss"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msepsq/trained_vgg_5_picture-instanthighlr_msepsq_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "loss_functions/"   
    elif tag=="msep2" or tag==2:
        title = "Parallel training with MSEp autoencoder loss (low lr)"
        ae_model =  home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture-instanthighlr_msep2/trained_vgg_5_picture-instanthighlr_msep2_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "loss_functions/"
        
    #-------------------Channel Encoders---------------------------
    elif tag=="channel_3n_noZeroEvent" or tag==3:
        title = "Parallel training with channel autoencoder (3 neurons) and balanced dataset"
        ae_model =  home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"
    elif tag=="channel_3n" or tag==4:
        title = "Parallel training with channel autoencoder (3 neurons)"
        ae_model =  home+"models/channel_3n_m3/trained_channel_3n_m3_autoencoder_test.txt" 
        prl_model = home+"models/channel_3n_m3/trained_channel_3n_m3_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"    
    elif tag=="channel_3n_noZero" or tag==5:
        title = "Parallel training with channel autoencoder (3 neurons) and no zero centering"
        ae_model =  home+"models/channel_3n_m3-noZero/trained_channel_3n_m3-noZero_autoencoder_test.txt" 
        prl_model = home+"models/channel_3n_m3-noZero/trained_channel_3n_m3-noZero_autoencoder_supervised_parallel_up_down_dropout_stateful_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"  
        
    elif tag=="channel_5n" or tag==6:
        title = "Parallel training with channel autoencoder (5 neurons)"
        ae_model =  home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/"
        epoch_schedule="1-1-1"
    elif tag=="channel_10n" or tag==7:
        title = "Parallel training with channel autoencoder (10 neurons)"
        ae_model =  home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_test.txt"
        prl_model = home+"models/channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_supervised_parallel_up_down_stateful_convdrop_test.txt"
        save_to_folder = "channel/" 
        epoch_schedule="1-1-1"
        
    #-------------------Up Down classifications vgg_5---------------------------
    elif tag=="vgg_3" or tag==8:
        title = "Parallel training with model '2000'"
        ae_model =  home+"models/vgg_3/trained_vgg_3_autoencoder_test.txt" 
        prl_model = home+"models/vgg_3/trained_vgg_3_autoencoder_supervised_parallel_up_down_test.txt"
        epoch_schedule="10-1-1"
        save_to_folder = "bottleneck/"
    
    elif tag=="vgg_5_600_picture" or tag==9:
        title = "Parallel training with model '600 picture'"
        ae_model =  home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_up_down_new_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_600_morefilter" or tag==10:
        title = "Parallel training with model '600 morefilter'"
        ae_model =  home+"models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_600_morefilter-new" or tag==40:
        title = "Parallel training with model '600 morefilter new'"
        ae_model =  home+"models/vgg_5_morefilter-new/trained_vgg_5_morefilter-new_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_morefilter-new/trained_vgg_5_morefilter-new_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_600_channel" or tag==11:
        title = "Parallel training with model '600 channel'"
        ae_model =  home+"models/vgg_5_channel/trained_vgg_5_channel_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_channel/trained_vgg_5_channel_autoencoder_supervised_parallel_up_down_dropout06_test.txt"
        print("60 % Dropout was used for this model. Others are available, but the overfitting is best seen here.")
        save_to_folder = "bottleneck/"
        
    elif tag=="vgg_5_200" or tag==12:
        title = "Parallel training with model '200'"
        ae_model =  home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_200_dense" or tag==13:
        title = "Parallel training with model '200 dense'"
        ae_model =  home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"  
    elif tag=="vgg_5_200_dense-new" or tag==41:
        title = "Parallel training with model '200 dense new'"
        ae_model =  home+"models/vgg_5_200_dense-new/trained_vgg_5_200_dense-new_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200_dense-new/trained_vgg_5_200_dense-new_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"  
        
    elif tag=="vgg_5_64" or tag==14:
        title = "Parallel training with model '64'"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_64-new" or tag==32:
        title = "Parallel training with model '64 new'"
        ae_model =  home+"models/vgg_5_64-new/trained_vgg_5_64-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64-new/trained_vgg_5_64-new_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
         
        
    elif tag=="vgg_5_32" or tag==15:
        title = "Parallel training with model '32' and $\epsilon=10^{-8}$"
        ae_model =  home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32/trained_vgg_5_32_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_32-eps01" or tag==16:
        title = "Parallel training with model '32'"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
    elif tag=="vgg_5_32-new" or tag==33:
        title = "Parallel training with model '32 new'"
        ae_model =  home+"models/vgg_5_32-new/trained_vgg_5_32-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-new/trained_vgg_5_32-new_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "bottleneck/"
        
    #-------------------vgg_5_600_picture Instant high lr Schmu---------------------------  
    elif tag=="vgg_5_600-ihlr" or tag==17:
        title = "Parallel training with model '600 picture' and high lr"
        ae_model =  home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_new_test.txt"
        save_to_folder = "instanthighlr/"
    elif tag=="vgg_5_600-ihlr_dense_deep" or tag==18:
        title = "Parallel training with model '600 picture', high lr and additional dense layer"
        ae_model =  home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_dense_deep_test.txt"
        save_to_folder = "instanthighlr/"
    elif tag=="vgg_5_600-ihlr_dense_shallow" or tag==19:
        title = "Parallel training with model '600 picture', high lr and removed dense layer"
        ae_model =  home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_dense_shallow_test.txt"
        save_to_folder = "instanthighlr/"
    elif tag=="vgg_5_600-ihlr_add_conv" or tag==20:
        title = "Parallel training with model '600 picture', high lr and additional convolutional layer"
        ae_model =  home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_add_conv_test.txt"
        save_to_folder = "instanthighlr/"
        
    #-------------------vgg_5_200 Parameter Erhöhung Up-down ---------------------------
    elif tag=="vgg_5_200_deep" or tag==21:
        title = "Parallel training with model '200 deep'"
        ae_model =  home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
    elif tag=="vgg_5_200_large" or tag==22:
        title = "Parallel training with model '200 large'"
        ae_model =  home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
    elif tag=="vgg_5_200_shallow" or tag==23:
        title = "Parallel training with model '200 shallow'"
        ae_model =  home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
    elif tag=="vgg_5_200_small" or tag==24:
        title = "Parallel training with model '200 small'"
        ae_model =  home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "vgg_5_200_params/"
        
    #------------------- Bottleneck energy ---------------------------------------------
    elif tag=="vgg_3_energy" or tag==25:
        title = "Parallel training with model '2000'"
        ae_model =  home+"models/vgg_3/trained_vgg_3_autoencoder_test.txt" 
        prl_model = home+"models/vgg_3/trained_vgg_3_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
        
    elif tag=="vgg_5_600_picture_energy" or tag==26:
        title = "Parallel training with model '600 picture'"
        ae_model =  home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_600_morefilter_energy" or tag==31:
        title = "Parallel training with model '600 morefilter'"
        ae_model =  home+"models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_600_morefilter_energy-new" or tag==42:
        title = "Parallel training with model '600 morefilter new'"
        ae_model =  home+"models/vgg_5_morefilter-new/trained_vgg_5_morefilter-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_morefilter-new/trained_vgg_5_morefilter-new_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
         
    elif tag=="vgg_5_200_energy" or tag==27:
        title = "Parallel training with model '200'"
        ae_model =  home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_energy_linear_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_200_dense_energy" or tag==30:
        title = "Parallel training with model '200 dense'"
        ae_model =  home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200_dense/trained_vgg_5_200_dense_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_200_dense_energy-new" or tag==43:
        title = "Parallel training with model '200 dense new'"
        ae_model =  home+"models/vgg_5_200_dense-new/trained_vgg_5_200_dense-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_dense-new/trained_vgg_5_200_dense-new_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
     
    elif tag=="vgg_5_64_energy" or tag==28:
        title = "Parallel training with model '64'"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_64_energy_nodrop" or tag==45:
        title = "Parallel training with model '64' no dropout"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_energy_drop00_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_64_energy-new" or tag==34:
        title = "Parallel training with model '64 new'"
        ae_model =  home+"models/vgg_5_64-new/trained_vgg_5_64-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64-new/trained_vgg_5_64-new_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
        
    elif tag=="vgg_5_32-eps01_energy" or tag==29:
        title = "Parallel training with model '32'"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_32-eps01_energy_nodrop" or tag==44:
        title = "Parallel training with model '32' no dropout"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_energy_drop00_test.txt"
        save_to_folder = "bottleneck_energy/" 
    elif tag=="vgg_5_32-eps01_energy-new" or tag==35:
        title = "Parallel training with model '32 new'"
        ae_model =  home+"models/vgg_5_32-new/trained_vgg_5_32-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-new/trained_vgg_5_32-new_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
        
    #-------------------vgg_5_200 Parameter Erhöhung Energy ---------------------------
    elif tag=="vgg_5_200_deep_energy" or tag==36:
        title = "Parallel training with model '200 deep'"
        ae_model =  home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "vgg_5_200_params_energy/"
    elif tag=="vgg_5_200_large_energy" or tag==37:
        title = "Parallel training with model '200 large'"
        ae_model =  home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_large/trained_vgg_5_200_large_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "vgg_5_200_params_energy/"
    elif tag=="vgg_5_200_shallow_energy" or tag==38:
        title = "Parallel training with model '200 shallow'"
        ae_model =  home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "vgg_5_200_params_energy/"
    elif tag=="vgg_5_200_small_energy" or tag==39:
        title = "Parallel training with model '200 small'"
        ae_model =  home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200_small/trained_vgg_5_200_small_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "vgg_5_200_params_energy/"
        
    #-------------------vgg_5_200 New Binning UpDown and Energy ---------------------------
    elif tag=="vgg_5_200-newbin" or tag==46:
        #title = "Parallel training with model '200 small'"
        ae_model =  home+"models/vgg_5_200-newbin/trained_vgg_5_200-newbin_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200-newbin/trained_vgg_5_200-newbin_autoencoder_supervised_parallel_up_down_test.txt"
        save_to_folder = "newbinning/"
        
    elif tag=="vgg_5_200-newbin_energy" or tag==47:
        #title = "Parallel training with model '200 small'"
        ae_model =  home+"models/vgg_5_200-newbin/trained_vgg_5_200-newbin_autoencoder_test.txt" 
        prl_model = home+"models/vgg_5_200-newbin/trained_vgg_5_200-newbin_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "newbinning/"
    
    #------------------- Broken Studies ---------------------------
    elif tag=="vgg_5_64-broken15_energy_broken15" or tag==48:
        ae_model =  home+"models/vgg_5_64-broken15/trained_vgg_5_64-broken15_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64-broken15/trained_vgg_5_64-broken15_autoencoder_supervised_parallel_energy_nodrop_broken15_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_64-broken15_energy" or tag==49:
        ae_model =  home+"models/vgg_5_64-broken15/trained_vgg_5_64-broken15_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64-broken15/trained_vgg_5_64-broken15_autoencoder_supervised_parallel_energy_nodrop_test.txt"
        save_to_folder = "bottleneck_energy/"
    
    #----------------------------------------------------------------------------------
    else:
        raise NameError("Tag "+str(tag)+" unknown.")
        
    test_files=[ae_model, prl_model]
    if printing==True: 
        print("Loaded the following files:")
        for file in test_files:
            print(file.split(home)[1]) 
           
    save_as=home+"results/plots/statistics/"+save_to_folder+"statistics_parallel_"+prl_model.split("/")[-1][:-4]+".pdf"
    return test_files, title, labels_override, save_as, epoch_schedule, style



def get_props_for_plot_parser(tag, printing=True):
    #For the script plots_statistics_parser
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    legend_locations=(1, "upper left")
    title=""
    #save it to this folder in the results/plots/ folder
    save_to_name = "statistics/statistics_parser_"+str(tag)+".pdf"
    #Colors to use. [] for auto selection
    colors=[]
    #Override xtick locations; None for automatic
    xticks=None
    #Default style:
    style="extended"
    #range for plot:
    xrange="auto"
    #Average over this many bins in the train data (to reduce jitter)
    average_train_data_bins=1

    try: tag=int(tag) 
    except: ValueError
    if tag=="channel-encs" or tag==0:
        title = "Encoder performance of channel autoencoder networks"
        test_files=["channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_epoch35_supervised_up_down_stateful_convdrop_test.txt", 
                    "channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_epoch17_supervised_up_down_stateful_convdrop_test.txt",
                    "channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_epoch23_supervised_up_down_stateful_convdrop_test.txt"]
        labels_override = ["3 neurons", "5 neurons", "10 neurons"]
        legend_locations=("lower right", "upper left")
        save_to_name = "statistics/statistics_parser_channel-encs.pdf"
        xrange=[0,50]
    
    elif tag=="pic_ihlr_enc_test" or tag==1:
        #vgg 5 picture ihlr: Parallel tests ob man den absturz der acc verhindern kann durch mehr dense layer (kann man nicht).
        title = "Variation of unfrozen encoder layers"
        test_files=["vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_new_test.txt",
                    "vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_add_conv_test.txt",
                    "vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_dense_deep_test.txt" ,
                    "vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_dense_shallow_test.txt",]
        labels_override = ["Two dense", "+Convolution", "Three dense", "One dense"]
        save_to_name = "statistics/statistics_parser_pic_ihlr_enc_test.pdf"
        
    elif tag=="unfreeze" or tag==2:
        title = "Successive unfreezing of encoder layers"
        test_files=["vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_test.txt" ,
                    "vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_broken4_test.txt",]
        labels_override = ["Normal dataset", "Manipulated dataset",]
        save_to_name = "unfreeze/broken4_vgg5_200_comp.pdf"
        colors = ["navy", "orange"]
        
    elif tag=="encoder_energy_new" or tag==3:
        title = "Different dropout rates for the encoder"
        test_files=["vgg_5_32-new/trained_vgg_5_32-new_autoencoder_epoch2_supervised_energy_dense_small_drop00_test.txt",
                    "vgg_5_32-new/trained_vgg_5_32-new_autoencoder_epoch2_supervised_energy_dense_small_drop01_test.txt",
                    "vgg_5_32-new/trained_vgg_5_32-new_autoencoder_epoch2_supervised_energy_dense_small_drop02_test.txt", 
                    "vgg_5_32-new/trained_vgg_5_32-new_autoencoder_epoch2_supervised_energy_dense_small_drop03_test.txt", 
                    "vgg_5_32-new/trained_vgg_5_32-new_autoencoder_epoch2_supervised_energy_dense_small_drop04_test.txt"]
        labels_override = ["0 %", "10 %", "20 %", "30 %", "40 %"]
        save_to_name = "statistics/statistics_parser_encoder_energy_new_test.pdf"
        xrange=[0,65]
        #colors = ["navy", "orange"]
    elif tag=="encoder_energy_drop" or tag==4:
        title = "Different dropout rates for the encoder"
        test_files=["vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_small_drop00_test.txt",
                    "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_small_drop01_test.txt",
                    "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_small_drop02_test.txt", 
                    "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_small_drop03_test.txt",]
        labels_override = ["0 %", "10 %", "20 %", "30 %"]
        save_to_name = "statistics/statistics_parser_encoder_energy_drop_test.pdf"
        xrange=[0,60]
        average_train_data_bins=8
        style="two_in_one_line"
    elif tag=="encoder_energy_size" or tag==5:
        title = "Size of first dense layer"
        test_files=["vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_verysmall_drop00_test.txt",
                    "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_small_drop00_test.txt",
                    "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_drop00_test.txt",
                    "vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_1000_drop00_test.txt",]
                    #"vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch18_supervised_energy_dense_4000_drop00_test.txt"]
        labels_override = ["32","64", "256", "1000",] #"4000"]
        save_to_name = "statistics/statistics_parser_encoder_energy_size_test.pdf"
        xrange=[0,60]
        style="two_in_one_line"
        average_train_data_bins=8
    
    elif tag=="vgg_3_encoder_drop" or tag==6:
        #title = "Size of first dense layer"
        test_files=["vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_nodrop_test.txt" ,
                    "vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_test.txt"]
        labels_override = ["No dropout", "20% dropout"]
        save_to_name = "statistics/statistics_parser_vgg_3_encoder_energy_drop_test.pdf"
        xrange=[0,50]
        style="two_in_one_line"
    
    
    
    
        
    #------------------------- Old setups, from plot_statistics_better -------------------------    
    
    elif tag=="vgg_3_comp":
        title="Progress of autoencoders with different optimization strategies"
        test_files = ["vgg_3/trained_vgg_3_autoencoder_test.txt",
              #"vgg_3_reg-e9/trained_vgg_3_reg-e9_autoencoder_test.txt",
              #"vgg_3-eps4/trained_vgg_3-eps4_autoencoder_test.txt",
              #"vgg_3_dropout/trained_vgg_3_dropout_autoencoder_test.txt",
              #"vgg_3_max/trained_vgg_3_max_autoencoder_test.txt",
              #"vgg_3_stride/trained_vgg_3_stride_autoencoder_test.txt",
              #"vgg_3_stride_noRelu/trained_vgg_3_stride_noRelu_autoencoder_test.txt",
              "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",]
              #"vgg_3_small/trained_vgg_3_small_autoencoder_test.txt",
              #"vgg_3_verysmall/trained_vgg_3_verysmall_autoencoder_test.txt",]
        labels_override = [r"Adam with $\epsilon=10^{-1}$", "SGD", r"Adam with $\epsilon=10^{-8}$"]
        save_to_name = "statistics/statistics_parser_vgg_3_comp_test.pdf"
    
    elif tag=="vgg4_autoencoders_var_depth":
        test_files = [#"vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
                  "vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
                  "vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
                  "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
                  "vgg_4_15c/trained_vgg_4_15c_autoencoder_test.txt",
                  "vgg_4_30c/trained_vgg_4_30c_autoencoder_test.txt",]
    
        title="Loss of autoencoders with a varying number of convolutional layers"
        xticks=[0,5,10,15,20,25,30]
        labels_override=["12 layers","14 layers", "16 layers", "20 layers", "30 layers", "60 layers"]
        
    
    elif tag=="vgg4_autoencoders_10c":
        test_files = [#"vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
                  #"vgg_4_ConvAfterPool/trained_vgg_4_ConvAfterPool_autoencoder_test.txt",
                  #"vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
                  #"vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
                  #"vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
                  "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
                  "vgg_4_10c_smallkernel/trained_vgg_4_10c_smallkernel_autoencoder_test.txt",
                  "vgg_4_10c_triple/trained_vgg_4_10c_triple_autoencoder_test.txt",
                  "vgg_4_10c_triple_same_structure/trained_vgg_4_10c_triple_same_structure_autoencoder_test.txt",]
                  #"vgg_4_7c_less_filters/trained_vgg_4_7c_less_filters_autoencoder_test.txt"]
        title="Loss of autoencoders with 20 convolutional layers"
        labels_override=["Standard", "Small kernel", "Triple structure", "Triple structure variation"]
        
    
        
    #range not adjustable currently, should be 0:50
    elif tag=="vgg_3_parallel_jumps":
        test_files=["vgg_3/trained_vgg_3_autoencoder_supervised_parallel_up_down_test.txt",]
        labels_override = [r"Adam with $\epsilon=10^{-1}$",]
        #title="Accuracy during parallel supervised training" 
        #xrange=[0,50]
        style="extended"#"two_in_one_line"
        colors=["yellow",]
    
    elif tag=="Unfrozen":
        test_files = ["vgg_3/trained_vgg_3_supervised_up_down_test.txt",
                  "vgg_3_dropout/trained_vgg_3_dropout_supervised_up_down_test.txt",
                  "vgg_3_max/trained_vgg_3_max_supervised_up_down_test.txt",
                  "vgg_3_stride/trained_vgg_3_stride_supervised_up_down_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_supervised_up_down_test.txt",]
    
    elif tag=="Encoders_Epoch_10":
        test_files = ["vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_test.txt",
                  "vgg_3_dropout/trained_vgg_3_dropout_autoencoder_epoch10_supervised_up_down_test.txt",
                  "vgg_3_max/trained_vgg_3_max_autoencoder_epoch10_supervised_up_down_test.txt",
                  "vgg_3_stride/trained_vgg_3_stride_autoencoder_epoch10_supervised_up_down_test.txt", ]
    
    elif tag=="sgdlr01_encoders":
        test_files = ["vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch2_supervised_up_down_test.txt",
                  "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch5_supervised_up_down_test.txt",
                  "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch10_supervised_up_down_test.txt",]
    
    elif tag=="Encoders_vgg_3_eps_AE_E10":
        test_files = ["vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_triplebatchnorm_e1_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_unfbatchnorm_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch80_supervised_up_down_unfbatchnorm_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_unfbatchnorm_no_addBN_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_batchnorm_e1_test.txt",
                  "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch11_supervised_up_down_test.txt",]
    
    elif tag=="vgg3eps": # AE E10 Encoders: finaler test von vgg_3
        test_files = [ "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_zero_center_and_norm_test.txt",
                   "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_BN_noBN_test.txt",
                   "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_norm_noBN_noDrop_BN_test.txt",
                   "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_downNorm_BN_noDrop_BN_test.txt",
                   "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_BN_BN_test.txt",
                   "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_noBN_BN_test.txt",
                   "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_zero_center_test.txt",]
    

        
    else:
        raise NameError("Tag "+str(tag)+" unknown.")
        
    if printing==True: 
        print("Loaded the following files:")
        for file in test_files:
            print(file) 
        
    test_files=[home+file for file in test_files]
    save_as="/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"+save_to_name
    return test_files, title, labels_override, save_as, legend_locations, colors, xticks, style, xrange, average_train_data_bins








def get_path_best_epoch(modeltag, full_path=True):
    #Get the path to the h5 file that had the best performance of a model
    #if full_path==True, will also include the base_path:
    base_path="/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    #------------------------- Up-Down Networks ------------------------- 
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #------------------------- unfrozen networks ------------------------
    if modeltag=="vgg_3_unf":
        model_path="vgg_3/trained_vgg_3_supervised_up_down_new_epoch5.h5"
        
    #------------------------- Bottleneck ------------------------- 
    elif modeltag=="vgg_3":
        model_path="vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_accdeg_epoch23.h5"
    
    elif modeltag=="vgg_5_600_picture" :
        model_path="vgg_5_picture/trained_vgg_5_picture_autoencoder_epoch48_supervised_up_down_epoch88.h5"
    elif modeltag=="vgg_5_600_morefilter":
        model_path=""
        raise
    
    elif modeltag=="vgg_5_200":
        model_path="vgg_5_200/trained_vgg_5_200_autoencoder_epoch94_supervised_up_down_epoch45.h5"
    elif modeltag=="vgg_5_200_dense":
        model_path=""
        raise
    
    elif modeltag=="vgg_5_64":
        model_path="vgg_5_64/trained_vgg_5_64_autoencoder_epoch64_supervised_up_down_epoch26.h5"
        
    elif modeltag=="vgg_5_32-eps01":
        model_path="vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_up_down_epoch43.h5"
    
    #------------------------------200 size variation------------------------------
    elif modeltag=="vgg_5_200_shallow":
        model_path="/vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_epoch19_supervised_up_down_epoch63.h5"
    elif modeltag=="vgg_5_200_small":
        model_path="vgg_5_200_small/trained_vgg_5_200_small_autoencoder_epoch77_supervised_up_down_epoch87.h5"
    elif modeltag=="vgg_5_200_large":
        model_path="vgg_5_200_large/trained_vgg_5_200_large_autoencoder_epoch39_supervised_up_down_epoch55.h5"
    elif modeltag=="vgg_5_200_deep":
        model_path="vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_epoch48_supervised_up_down_epoch60.h5"
    
    
    #------------------------------Unfreeze Networks------------------------------
    elif modeltag=="vgg_5_200-unfreeze_contE20":
        model_path="vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_contE20_epoch30.h5"
    
    
    
    #------------------------- Energy Networks ------------------------- 
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #---------------- Single unfrozen datafiles ----------------
    elif modeltag=="2000_unf_E":
        model_path = "vgg_5_2000/trained_vgg_5_2000_supervised_energy_epoch17.h5"
    elif modeltag=="2000_unf_mse_E":
        model_path = "vgg_5_2000-mse/trained_vgg_5_2000-mse_supervised_energy_epoch10.h5"
    elif modeltag=="200_linear_E":
        model_path="vgg_5_200/trained_vgg_5_200_autoencoder_supervised_parallel_energy_linear_epoch18.h5"
    
    
    #------------------------------Energy bottleneck------------------------------
    elif modeltag=="vgg_3_2000_E":
        model_path="vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_init_epoch29.h5"
    elif modeltag=="vgg_3_2000_E_nodrop":
        model_path="vgg_3/trained_vgg_3_autoencoder_epoch8_supervised_energy_nodrop_epoch7.h5"
        
    elif modeltag=="vgg_5_600_picture_E_nodrop":
        model_path="vgg_5_picture/trained_vgg_5_picture_autoencoder_epoch44_supervised_energy_nodrop_epoch14.h5"
    elif modeltag=="vgg_5_600_morefilter_E_nodrop":
        model_path="vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_epoch43_supervised_energy_nodrop_epoch8.h5"
        
    elif modeltag=="vgg_5_200_E_nodrop":
        model_path="vgg_5_200/trained_vgg_5_200_autoencoder_epoch94_supervised_energy_nodrop_epoch52.h5"
    elif modeltag=="vgg_5_200_dense_E_nodrop":
        model_path="vgg_5_200_dense-new/trained_vgg_5_200_dense-new_autoencoder_epoch101_supervised_energy_nodrop_epoch29.h5"
    
    elif modeltag=="vgg_5_64_E_nodrop":
        model_path="vgg_5_64/trained_vgg_5_64_autoencoder_epoch78_supervised_energy_drop00_2_epoch55.h5"
        
    elif modeltag=="vgg_5_32_E_nodrop":
        model_path="vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_epoch44_supervised_energy_nodrop_epoch90.h5"

    #------------------------------200 size variation------------------------------
    elif modeltag=="vgg_5_200_shallow_E_nodrop":
        model_path="vgg_5_200_shallow/trained_vgg_5_200_shallow_autoencoder_epoch5_supervised_energy_nodrop_epoch24.h5"
    elif modeltag=="vgg_5_200_small_E_nodrop":
        model_path="vgg_5_200_small/trained_vgg_5_200_small_autoencoder_epoch58_supervised_energy_nodrop_epoch49.h5"
    elif modeltag=="vgg_5_200_large_E_nodrop":
        model_path="vgg_5_200_large/trained_vgg_5_200_large_autoencoder_epoch45_supervised_energy_nodrop_epoch5.h5"
    elif modeltag=="vgg_5_200_deep_E_nodrop":
        model_path="vgg_5_200_deep/trained_vgg_5_200_deep_autoencoder_epoch41_supervised_energy_nodrop_epoch31.h5"
        
    else: raise NameError("Tag '"+str(modeltag)+"' is not known!")
    
    if full_path:
        model_path=base_path+model_path
    return model_path



def get_highest_tagnumbers():
    tag_no=0
    while True:
        try:
            get_props_for_plot_parallel(tag_no, printing=False)
        except NameError: 
            parallel_tag_no=tag_no
            break
        tag_no+=1
    
    tag_no=0
    while True:
        try:
            get_props_for_plot_parser(tag_no, printing=False)
        except NameError: 
            parser_tag_no=tag_no
            break
        tag_no+=1
    
    print("\nHighest parallel tag no: "+str(parallel_tag_no-1))
    print("Highest parser tag no: "+str(parser_tag_no-1))


if __name__=="__main__":
    get_highest_tagnumbers()


"""
From the old plot_statistics_better script:
    
test_files = ["vgg_3/trained_vgg_3_autoencoder_test.txt",
              #"vgg_3_reg-e9/trained_vgg_3_reg-e9_autoencoder_test.txt",
              #"vgg_3-eps4/trained_vgg_3-eps4_autoencoder_test.txt",
              #"vgg_3_dropout/trained_vgg_3_dropout_autoencoder_test.txt",
              #"vgg_3_max/trained_vgg_3_max_autoencoder_test.txt",
              #"vgg_3_stride/trained_vgg_3_stride_autoencoder_test.txt",
              #"vgg_3_stride_noRelu/trained_vgg_3_stride_noRelu_autoencoder_test.txt",
              "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",]
              #"vgg_3_small/trained_vgg_3_small_autoencoder_test.txt",
              #"vgg_3_verysmall/trained_vgg_3_verysmall_autoencoder_test.txt",]

#Unfrozen
xtest_files = ["vgg_3/trained_vgg_3_supervised_up_down_test.txt",
              "vgg_3_dropout/trained_vgg_3_dropout_supervised_up_down_test.txt",
              "vgg_3_max/trained_vgg_3_max_supervised_up_down_test.txt",
              "vgg_3_stride/trained_vgg_3_stride_supervised_up_down_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_supervised_up_down_test.txt",]

#Encoders Epoch 10
xtest_files = ["vgg_3/trained_vgg_3_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_dropout/trained_vgg_3_dropout_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_max/trained_vgg_3_max_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_stride/trained_vgg_3_stride_autoencoder_epoch10_supervised_up_down_test.txt", ]

#sgdlr01 encoders
xtest_files = ["vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch2_supervised_up_down_test.txt",
              "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch5_supervised_up_down_test.txt",
              "vgg_3-sgdlr01/trained_vgg_3-sgdlr01_autoencoder_epoch10_supervised_up_down_test.txt",]

#Enocders vgg_3_eps AE E10
xtest_files = ["vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_triplebatchnorm_e1_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_unfbatchnorm_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch80_supervised_up_down_unfbatchnorm_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_unfbatchnorm_no_addBN_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_batchnorm_e1_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch11_supervised_up_down_test.txt",]

#vgg3eps AE E10 Encoders: finaler test von vgg_3
xtest_files = [ "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_zero_center_and_norm_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_BN_noBN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_norm_noBN_noDrop_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_downNorm_BN_noDrop_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_BN_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_noNorm_noBN_BN_test.txt",
               "vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_zero_center_test.txt",]

#only
xtest_files = ["vgg_5_picture/trained_vgg_5_picture_autoencoder_supervised_parallel_up_down_new_test.txt",]

#vgg4 autoencoders variational depth
xtest_files = [#"vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
              "vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
              "vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
              "vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
              "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
              "vgg_4_15c/trained_vgg_4_15c_autoencoder_test.txt",
              "vgg_4_30c/trained_vgg_4_30c_autoencoder_test.txt",]

xtitle="Loss of autoencoders with a varying number of convolutional layers"
xticks=[0,5,10,15,20,25,30]
xlabels_override=["12 layers","14 layers", "16 layers", "20 layers", "30 layers", "60 layers"]


#vgg4 autoencoders 10c tests
test_files = [#"vgg_3_eps/trained_vgg_3_eps_autoencoder_test.txt",
              #"vgg_4_ConvAfterPool/trained_vgg_4_ConvAfterPool_autoencoder_test.txt",
              #"vgg_4_6c/trained_vgg_4_6c_autoencoder_test.txt",
              #"vgg_4_6c_scale/trained_vgg_4_6c_scale_autoencoder_test.txt",
              #"vgg_4_8c/trained_vgg_4_8c_autoencoder_test.txt",
              "vgg_4_10c/trained_vgg_4_10c_autoencoder_test.txt",
              "vgg_4_10c_smallkernel/trained_vgg_4_10c_smallkernel_autoencoder_test.txt",
              "vgg_4_10c_triple/trained_vgg_4_10c_triple_autoencoder_test.txt",
              "vgg_4_10c_triple_same_structure/trained_vgg_4_10c_triple_same_structure_autoencoder_test.txt",]
              #"vgg_4_7c_less_filters/trained_vgg_4_7c_less_filters_autoencoder_test.txt"]
title="Loss of autoencoders with 20 convolutional layers"
labels_override=["Standard", "Small kernel", "Triple structure", "Triple structure variation"]

#vgg_5 smaller bottleneck
xtest_files = ["vgg_5_channel/trained_vgg_5_channel_autoencoder_test.txt",
              "vgg_5_picture/trained_vgg_5_picture_autoencoder_test.txt",
              "vgg_3/trained_vgg_3_autoencoder_test.txt",
              "vgg_5_morefilter/trained_vgg_5_morefilter_autoencoder_test.txt",]
"""

