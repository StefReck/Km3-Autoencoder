# -*- coding: utf-8 -*-
"""
For every epoch of a model, calculate the up-down robustness, and log it into a file.
For the robustness, the performance of the model has to be evaluated 3 times:
    Broken model  auf Broken
    Broken model  auf real
    Real model    auf real
"""

from evaluation_dataset import print_statistics_in_numbers
from util.evaluation_utilities import make_or_load_files
import os

#pathes to the models
model_base_ae = "models/vgg_5_200-unfreeze/trained_vgg_5_200-unfreeze_autoencoder_epoch1_"
model_base_real =  model_base_ae + "supervised_up_down_"
model_base_broken= model_base_ae + "supervised_up_down_broken4_"
#Datasets to test on
dataset_broken  = "xzt_broken4"
dataset_real    = "xzt"

#Procedure:         model to use, dataset to test on
procedure = ( (model_base_broken, dataset_broken),  #on simulations
              (model_base_broken, dataset_real),    #on measured data
              (model_base_real,   dataset_real), )  #upperlim on measured data

#Starting epoch
epoch=1
#bins for the histogramm of which the robutness is calculated
bins=32
#Class type
class_type=(2,"up_down")

#Name of the file the lines will be logged to, generated automatically
name_of_logfile = model_base_ae + "unfreeze_broken4_log.txt"


if not os.path.isfile(name_of_logfile):
    new_logfile=True
else:
    new_logfile=False
    
    
while True:
    print("\nWorking on epoch", epoch)
    
    modelidents, dataset_array = [],[]
    for modelbase, dataset in procedure:
        model_file = modelbase + "epoch"+str(epoch)+".h5"
        if not os.path.isfile(model_file):
            print("Did not find the file", model_file, ", exiting...")
            break 
        modelidents.append(model_file)
        dataset_array.append(dataset)
        
    #stats is Total acc for up_down class type of every model
    hist_data_array, stats_array = make_or_load_files(modelidents, dataset_array, bins, class_type=class_type, also_return_stats=True)
    #header/line contains: (Sim-Meas)/Meas\t(Upperlim-Meas)/Meas
    header, line = print_statistics_in_numbers(hist_data_array, plot_type="acc", return_line=True)
    
    #What will be logged to a file, seperated by tabs
    stuff_to_log_head = ["#epoch",  header[0],header[1],"acc_sim",          "acc_meas",         "acc_ulim"]
    stuff_to_log_data = [str(epoch),line[0],  line[1],  str(stats_array[0]),str(stats_array[1]),str(stats_array[2])]
    
    logheader, logline = stuff_to_log_head[0], "\n"+stuff_to_log_data[0]
    for i in range(1,len(stuff_to_log_head)):
        logheader += "\t" + stuff_to_log_head[i][:10]
        logline   += "\t" + stuff_to_log_data[i][:10]

    with open(name_of_logfile, "a") as logfile:
        if new_logfile:
            logfile.write(logheader)
        logfile.write(logline)
        print("Wrote into logfile")
        
    epoch+=1

print("\nDone, logfile is at", name_of_logfile)
    
    