# -*- coding: utf-8 -*-

import matplotlib
from keras.models import load_model, Model
import h5py
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.backends.backend_pdf import PdfPages
import argparse


def parse_input():
    parser = argparse.ArgumentParser(description='Make histograms of models, and save them to the same directory.')
    parser.add_argument('model', type=str, help='The model of which to do a histogram.')

    args = parser.parse_args()
    params = vars(args)
    return params

#params = parse_input()
#model_name_and_path = params["model"]

#Debug on Laptop:
model_name_and_path="../Daten/xzt/trained_vgg_3_eps_autoencoder_epoch10.h5"



save_plot_as=model_name_and_path[:-3]
model=load_model(model_name_and_path)

#Data from which to take the events
#for xzt
data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
#train_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
data=data_path+test_data
zero_center = data_path+zero_center_data
    


#On Laptop:
data = "../Daten/xzt/JTE_KM3Sim_gseagen_elec-CC_3-100GeV-1_1E6-1bin-3_0gspec_ORCA115_9m_2016_100_xzt.h5"
zero_center = "../Daten/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
"""
model_eps = "Daten/xzt/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_epoch10.h5"
model = "Daten/xzt/trained_vgg_3_autoencoder_epoch10_supervised_up_down_epoch10.h5"
model_sup = "Daten/xzt/trained_vgg_3_supervised_up_down_epoch3.h5"
autoencoder_model = "Daten/xzt/trained_vgg_3_eps_autoencoder_epoch10.h5"

encoder = load_model(model)
encoder_eps = load_model(model_eps)
encoder_sup = load_model(model_sup)
autoencoder = load_model(autoencoder_model)
"""

#Which event(s) should be taken from the file to make the histogramms
which_events = [0,]


file=h5py.File(data , 'r')
zero_center_image = np.load(zero_center)
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
labels = file["y"][which_events] 
hists = file["x"][which_events]
#Get some hists from the file
hists=hists.reshape((hists.shape+(1,))).astype(np.float32)
#0 center them
centered_hists = np.subtract(hists, zero_center_image)



#Predict on 0 centered data
#pred=encoder.predict_on_batch(centered_hists)
#pred_eps=encoder_eps.predict_on_batch(centered_hists)


def get_out_from_layer(layer_no, model):
    get_layer_1_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_no].output])
    layer_1_output = get_layer_1_output([centered_hists,0])[0]
    return layer_1_output

def make_histogramms_of_layer(layer_no, model_1, model_2=None, title_1="Epsilon = 0.1", title_2="Epsilon = E-8"):
    #histogram of layer outputs
    enc_feat=get_out_from_layer(layer_no, model_1)
    enc_eps_feat=get_out_from_layer(layer_no, model_2) if model_2 is not None else None
    
    plt.figure()
    
    if model_2 is not None:
        plt.subplot(121)
        plt.title(title_1)
        plt.hist(enc_feat.flatten(), 100)
        
        plt.subplot(122)
        plt.title(title_2)
        plt.hist(enc_eps_feat.flatten(), 100)
        
        plt.suptitle(model_1.layers[layer_no].name)
    
    else:
        plt.title(model_1.layers[layer_no].name)
        plt.hist(enc_feat.flatten(), 100)
        
    plt.tight_layout()

def make_histogramms_of_layer_channel(layer_no, model_1):
    #histogram of layer outputs, every channel in its own color
    enc_feat=get_out_from_layer(layer_no, model_1)#(1,4,6,10,64)
    enc_feat=enc_feat.reshape(-1, enc_feat.shape[-1]) #240,64
    
    plt.figure()
    matplotlib.rcParams.update({'font.size': 12})
    plt.title('Output of last Batch Normalization') #Autoencoder $\epsilon = 10^{-8}$ after Epoch 10: 
    repeats_of_colormap=4
    color=cm.rainbow(np.linspace(0,1,int(enc_feat.shape[-1]/repeats_of_colormap)))
    color=np.tile(color,(repeats_of_colormap,1))
    plt.hist(enc_feat, 100, stacked=True, color=color, zorder=10)
    plt.xlabel("Output of neuron")
    plt.ylabel("Number of neurons")
    plt.grid(zorder=0)
    plt.tight_layout()


def make_weights_histogramms_of_layer(layer_no, model_1):
    #histogram of weights
    weights = []
    for w in model_1.layers[layer_no].get_weights():
        weights.extend(w.flatten())

    plt.hist(weights, 100)
    plt.title(model_1.layers[layer_no].name)

#Go through whole model and make histogramm of output of every layer
#Can plot one model with auto title, or two side by side, each with its own title and auto suptitle
def make_complete_prop(model_1, save_path, model_2=None, title_1="Epsilon = 0.1", title_2 = "Epsilon = E-8"):
    with PdfPages(save_path) as pp:
        for i in range(0,len(model_1.layers)):
            print("Generating output histogram of layer ",i, "of ", len(model_1.layers))
            make_histogramms_of_layer(i, model_1, model_2, title_1, title_2)
            pp.savefig()
            plt.close()

def make_complete_prop_weights(model_1, save_path):
    with PdfPages(save_path) as pp:
        for i in range(0,len(model_1.layers)):
            print("Generating weights histogram of layer ",i, "of ", len(model_1.layers))
            make_weights_histogramms_of_layer(i, model_1)
            pp.savefig()
            plt.close()
            
#make_complete_prop(model_1 = encoder, model_2=encoder_eps, title_1="Epsilon = 0.1", title_2 = "Epsilon = E-8", save_path="vgg_3_eps_autoencoder_epoch10_supervised_up_down_epoch10_activation_1_event.pdf")
make_histogramms_of_layer_channel(24, model)
      
        
#make_complete_prop(model, save_plot_as+"_layer_output.pdf")
#make_complete_prop_weights(model, save_plot_as+"_weights.pdf")




