# -*- coding: utf-8 -*-
"""
Some saved setups for the plot ststistics scripts.
"""
import numpy as np

def get_props_for_plot_parallel(tag):
    #For the script plots_statistics_parallel, which takes exactly two models
    #as an input (AE and parallel encoder)
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/"
    epoch_schedule="10-2-1"
    labels_override = ["Autoencoder", "Encoder"] 
    save_to_folder = ""
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
        
    #-------------------vgg_5_200 Parameter Erhöhung---------------------------
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
        
    #-------------------Energy reconstructions---------------------------
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
        
    elif tag=="vgg_5_64_energy" or tag==28:
        title = "Parallel training with model '64'"
        ae_model =  home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64/trained_vgg_5_64_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_64_energy_new" or tag==34:
        title = "Parallel training with model '64 new'"
        ae_model =  home+"models/vgg_5_64-new/trained_vgg_5_64-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_64-new/trained_vgg_5_64-new_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
        
    elif tag=="vgg_5_32-eps01_energy" or tag==29:
        title = "Parallel training with model '32'"
        ae_model =  home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-eps01/trained_vgg_5_32-eps01_autoencoder_supervised_parallel_energy_test.txt"
        save_to_folder = "bottleneck_energy/"
    elif tag=="vgg_5_32-eps01_energy_new" or tag==35:
        title = "Parallel training with model '32 new'"
        ae_model =  home+"models/vgg_5_32-new/trained_vgg_5_32-new_autoencoder_test.txt"
        prl_model = home+"models/vgg_5_32-new/trained_vgg_5_32-new_autoencoder_supervised_parallel_energy_test.txt"
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


def get_props_for_plot_parser(tag):
    #For the script plots_statistics_parser
    home = "/home/woody/capn/mppi013h/Km3-Autoencoder/models/"
    legend_locations=(1, "upper left")
    #save it to this folder in the results/plots/ folder
    save_to_name = "statistics/statistics_parser_"+tag+".pdf"
    #Colors to use. [] for auto selection
    colors=[]
    #Override xtick locations; None for automatic
    xticks=None

    if tag=="channel-encs":
        title = "Encoder performance of channel autoencoder networks"
        test_files=["channel_3n_m3-noZeroEvent/trained_channel_3n_m3-noZeroEvent_autoencoder_epoch35_supervised_up_down_stateful_convdrop_test.txt", 
                    "channel_5n_m3-noZeroEvent/trained_channel_5n_m3-noZeroEvent_autoencoder_epoch17_supervised_up_down_stateful_convdrop_test.txt",
                    "channel_10n_m3-noZeroEvent/trained_channel_10n_m3-noZeroEvent_autoencoder_epoch23_supervised_up_down_stateful_convdrop_test.txt"]
        labels_override = ["3 neurons", "5 neurons", "10 neurons"]
        legend_locations=("lower right", "upper left")
    
    elif tag=="pic_ihlr_enc_test":
        #vgg 5 picture ihlr: Parallel tests ob man den absturz der acc verhindern kann durch mehr dense layer (kann man nicht).
        title = "Variation of unfrozen encoder layers"
        test_files=["vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_new_test.txt",
                    "vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_add_conv_test.txt",
                    "vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_dense_deep_test.txt" ,
                    "vgg_5_picture-instanthighlr/trained_vgg_5_picture-instanthighlr_autoencoder_supervised_parallel_up_down_dense_shallow_test.txt",]
        labels_override = ["Two dense", "+Convolution", "Three dense", "One dense"]
        
    elif tag=="unfreeze":
        title = "Successive unfreezing of encoder layers"
        test_files=["vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_test.txt" ,
                    "vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_supervised_up_down_broken4_test.txt",]
        labels_override = ["Normal dataset", "Manipulated dataset",]
        save_to_name = "unfreeze/broken4_vgg5_200_comp.pdf"
        colors = ["navy", "orange"]
        
        
        
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
        
    
        """
    #range not adjustable currently, should be 0:50
    elif tag=="vgg_3_parallel_jumps":
        test_files=["models/vgg_3/trained_vgg_3_autoencoder_supervised_parallel_up_down_test.txt",]
        labels_override = [r"Adam with $\epsilon=10^{-1}$",]
        title="Accuracy during parallel supervised training" 
    """
    
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
        raise NameError("Tag "+tag+" unknown.")
        
    print("Loaded the following files:")
    for file in test_files:
        print(file) 
        
    test_files=[home+file for file in test_files]
    
    save_as="/home/woody/capn/mppi013h/Km3-Autoencoder/results/plots/"+save_to_name
    return test_files, title, labels_override, save_as, legend_locations, colors, xticks



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

