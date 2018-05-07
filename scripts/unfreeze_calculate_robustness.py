# -*- coding: utf-8 -*-
"""
For every epoch of a model, calculate the up-down robustness and a bunch of other infos,
and log it into a file. The logfile will contain:
    epoch, (Sim-Meas)/Meas, (Upperlim-Meas)/Meas ,"acc_sim", "acc_meas", "acc_ulim"

For the robustness, the performance of the model has to be evaluated 3 times:
    Broken model  auf Broken    (what you see when developing the model, acc_sim)
    Broken model  auf real      (the actual performance on proper data, acc_meas)
    Real model    auf real      (the best-case scenario of what the network can get, acc_ulim)
The above acc values are on a dataset basis (average over all events in that set)

Robustness indicators:
    (acc_sim -acc_meas)/acc_meas    : How much worse is the model on actual data
    (acc_ulim-acc_meas)/acc_meas    : How much worse is it then the best case
For these, the performance is binned energywise before calculating, so this is more 
of a per model basis (low energy events not more important then others).
"""

from evaluation_dataset import print_statistics_in_numbers
from util.evaluation_utilities import make_or_load_files
import os

tag="200_broken4"

def make_setup(tag):
    if tag=="200_broken4":
        #pathes to the models
        model_base_ae = "models/vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_"
        model_base_real =  model_base_ae + "supervised_up_down_"
        model_base_broken= model_base_ae + "supervised_up_down_broken4_"
        #Datasets to test on
        dataset_broken  = "xzt_broken4"
        dataset_real    = "xzt"
        
        #Class type
        class_type=(2,"up_down")
        #Name of the file the lines will be logged to, generated automatically
        name_of_logfile = "results/unfreeze_plot_data/"+model_base_ae.split("/")[-1] + "unfreeze_broken4_log.txt"
        #Starting epoch
        epoch=1
        #bins for the histogramm of which the robutness is calculated
        bins=32
        
    elif tag=="200_broken4_contE20":
        #From the above training at E20 (with 3 unfrozen C layers)
        model_base_ae = "models/vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_"
        model_base_real =  model_base_ae + "supervised_up_down_contE20_"
        model_base_broken= model_base_ae + "supervised_up_down_contE20_broken4_"
        #Datasets to test on
        dataset_broken  = "xzt_broken4"
        dataset_real    = "xzt"
        
        #Class type
        class_type=(2,"up_down")
        #Name of the file the lines will be logged to, generated automatically
        name_of_logfile = "results/unfreeze_plot_data/"+model_base_ae.split("/")[-1] + "unfreeze_contE20_broken4_log.txt"
        #Starting epoch
        epoch=1
        #bins for the histogramm of which the robutness is calculated
        bins=32
        
    else: raise NameError
    
    return model_base_real, model_base_broken, dataset_real, dataset_broken, class_type, name_of_logfile, epoch, bins

def calculate_robustness_logfile(model_base_real, model_base_broken, 
                         dataset_real, dataset_broken,
                         class_type, name_of_logfile,
                         epoch=1, bins=32):

    #Procedure:         model to use, dataset to test on
    procedure = ((model_base_broken, dataset_broken),  #on simulations
                 (model_base_broken, dataset_real),    #on measured data
                 (model_base_real,   dataset_real), )  #upperlim on measured data

    
    if not os.path.isfile(name_of_logfile):
        new_logfile=True
    else:
        new_logfile=False
        
    print("Logfile:", name_of_logfile)
    while True:
        print("\nWorking on epoch", epoch)
        
        #Get the three model+dataset combos for the current epoch in a list
        modelidents, dataset_array = [],[]
        for modelbase, dataset in procedure:
            model_file = modelbase + "epoch"+str(epoch)+".h5"
            if not os.path.isfile(model_file):
                print("Did not find the model", model_file, ", exiting...")
                break 
            modelidents.append(model_file)
            dataset_array.append(dataset)
            
        #stats is Total acc for up_down class type of every model
        hist_data_array, stats_array = make_or_load_files(modelidents, dataset_array, bins, class_type=class_type, also_return_stats=True)
        #header/line contains: (Sim-Meas)/Meas\t(Upperlim-Meas)/Meas
        header, line = print_statistics_in_numbers(hist_data_array, plot_type="acc", return_line=True)
        
        #What will be logged to a file, seperated by tabs
        stuff_to_log_head = ["#epoch",  header[0],   header[1],    "acc_sim",         "acc_meas",         "acc_ulim"]
        stuff_to_log_data = [str(epoch),str(line[0]),str(line[1]), str(stats_array[0]),str(stats_array[1]),str(stats_array[2])]
        
        logheader, logline = stuff_to_log_head[0], "\n"+stuff_to_log_data[0]
        for i in range(1,len(stuff_to_log_head)):
            logheader += "\t" + stuff_to_log_head[i][:10]
            logline   += "\t" + stuff_to_log_data[i][:10]
    
        with open(name_of_logfile, "a") as logfile:
            if new_logfile:
                logfile.write(logheader)
                new_logfile=False
            logfile.write(logline)
            print("Wrote into logfile")
            
        epoch+=1
    
    print("\nDone, logfile is at", name_of_logfile)
        
calculate_robustness_logfile(*make_setup(tag))
    
    