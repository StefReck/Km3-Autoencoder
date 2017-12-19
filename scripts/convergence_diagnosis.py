# -*- coding: utf-8 -*-
"""
Create layer output histograms of the last layers of a network, while training.
"""
from keras.layers import Activation, Input, Dropout, Dense, Flatten, Conv3D, MaxPooling3D, UpSampling3D,BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D
from keras.models import load_model, Model
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import h5py
from keras import optimizers
from matplotlib.backends.backend_pdf import PdfPages

from util.run_cnn import generate_batches_from_hdf5_file

plot_after_how_many_batches=[1,2,3,10,15,20,25,30,40,50,60,100,200,300,400,500] #0 is automatically plotted
#Which event(s) should be taken from the test file to make the histogramms
which_events = np.arange(0,100)
name_of_plots="vgg_3_autoencoder_eps_epoch10_convergence_analysis" #added will be : _withBN.pdf

laptop=False
if laptop == True:
    #autoencoder:
    model_name_and_path="Daten/xzt/trained_vgg_3_eps_autoencoder_epoch10.h5"
    #Data to produce from:
    data = "Daten/xzt/JTE_KM3Sim_gseagen_elec-CC_3-100GeV-1_1E6-1bin-3_0gspec_ORCA115_9m_2016_100_xzt.h5"
    zero_center = "Daten/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
    test_data=data
else:
    #autoencoder:
    model_name_and_path="/home/woody/capn/mppi013h/Km3-Autoencoder/models/vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch10.h5"
    #Data to produce from:
    #for xzt
    data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
    train_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
    test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
    
    data=data_path+train_data
    zero_center=data_path+zero_center_data
    test_data=data_path+test_data


def model_setup(autoencoder_model):
    autoencoder = load_model(autoencoder_model)
    #setup the vgg_3 model for convergence analysis
    def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis, strides=(1,1,1), dropout=0.0, ac_reg_penalty=0):
        regular = regularizers.l2(ac_reg_penalty) if ac_reg_penalty is not 0 else None
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable, activity_regularizer=regular)(inp)
        x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
        x = Activation('relu', trainable=trainable)(x)
        if dropout > 0.0: x = Dropout(dropout)(x)
        return x
    def setup_vgg_3(autoencoder, with_batchnorm):
        #832k params
        train=False
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        
        inputs = Input(shape=(11,18,50,1))
        x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
        x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
        
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
        x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
        x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
        
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
        x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
        encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
        encoder= Model(inputs=inputs, outputs=encoded)
        for i,layer in enumerate(encoder.layers):
            layer.set_weights(autoencoder.layers[i].get_weights())
        
        x = Flatten()(encoded)
        if with_batchnorm == True:
            x = BatchNormalization(axis=channel_axis)(x) 
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x) #init: std 0.032 = sqrt(2/1920), mean=0
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model_noBN = setup_vgg_3(autoencoder, with_batchnorm=False)
    model_noBN.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model_withBN = setup_vgg_3(autoencoder, with_batchnorm=True)
    model_withBN.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model_noBN, model_withBN
    
def make_histogramms_of_4_layers(centered_hists, layer_no_array, model_1, suptitle, title_array):
    #histograms of outputs of 4 layers in a 2x2 plot
    
    def get_out_from_layer(layer_no, model, centered_hists):
        get_layer_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_no].output])
        layer_output = get_layer_output([centered_hists,0])[0]
        return layer_output
    
    fig = plt.figure(figsize=(8,8))
    
    for i,layer_no in enumerate(layer_no_array):
        enc_feat=get_out_from_layer(layer_no, model_1, centered_hists)
        plt.subplot(221+i)
        plt.title(title_array[i])
        plt.hist(enc_feat.flatten(), 100)
    
    plt.suptitle(suptitle)
    #plt.tight_layout()
    return fig

def generate_plots(model_noBN, model_withBN, centered_hists, suptitles=["Layer outputs without batch normalization", "Layer outputs with batch normalization"]):
    title_array1=["Output from frozen encoder", "First dense layer", "Second dense layer", "Third dense layer"]
    title_array2=["Batch normalization", "First dense layer", "Second dense layer", "Third dense layer"]
    fig_noBN = make_histogramms_of_4_layers(centered_hists, [-4,-3,-2,-1], model_noBN, suptitle=suptitles[0], title_array=title_array1)
    fig_withBN = make_histogramms_of_4_layers(centered_hists, [-4,-3,-2,-1], model_withBN, suptitle=suptitles[1], title_array=title_array2)
    return fig_noBN, fig_withBN
    

#Generate 0-centered histogramms:
file=h5py.File(test_data , 'r')
zero_center_image = np.load(zero_center)
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
labels = file["y"][which_events] 
hists = file["x"][which_events]
#Get some hists from the file
hists=hists.reshape((hists.shape+(1,))).astype(np.float32)
#0 center them
centered_hists = np.subtract(hists, zero_center_image)


model_noBN, model_withBN = model_setup(autoencoder_model=model_name_and_path)

def convergence_analysis(model_noBN, model_withBN, centered_hists, plot_after_how_many_batches, data_for_generator):
    history_noBN=[]
    history_withBN=[]
    figures=[]
    
    fig1, fig2 = generate_plots(model_noBN, model_withBN, centered_hists, suptitles=["Layer outputs without batch normalization (0 batches)", "Layer outputs with batch normalization (0 batches)"])
    figures.append([0,fig1,fig2])
    
    generator = generate_batches_from_hdf5_file(filepath=data_for_generator, batchsize=32, n_bins=(11,18,50,1), class_type=(2, 'up_down'), is_autoencoder=False)
    i=0
    while i < max(plot_after_how_many_batches):
        x,y = next(generator)
        history_noBN.append(model_noBN.train_on_batch(x, y))
        history_withBN.append(model_withBN.train_on_batch(x, y))
        i+=1
        
        if i in plot_after_how_many_batches:
            fig1, fig2 = generate_plots(model_noBN, model_withBN, centered_hists, suptitles=["Layer outputs without batch normalization ("+str(i)+" batches)", "Layer outputs with batch normalization ("+str(i)+" batches)"])
            figures.append([i,fig1,fig2])
        
        
    return history_noBN, history_withBN, figures

def save_to_pdf(figures, name_of_plots="Test"):
    with PdfPages(name_of_plots+ "_noBN.pdf") as pp_noBN:
        with PdfPages(name_of_plots+ "_withBN.pdf") as pp_withBN:
            for tupel in figures:
                pp_noBN.savefig(tupel[1])
                plt.close()
                pp_withBN.savefig(tupel[2])
                plt.close()

#figures: [ int number of batches this plot was made after, figure no BN, figure w BN]
history_noBN, history_withBN, figures = convergence_analysis(model_noBN, model_withBN, centered_hists, plot_after_how_many_batches, data)
print("No BN:", history_noBN)
print("With BN:", history_withBN)
save_to_pdf(figures, name_of_plots)



