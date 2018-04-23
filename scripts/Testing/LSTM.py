# -*- coding: utf-8 -*-

"""
Test of the LSTM setup
The network is fed from 4D histograms (x,y,z,t).
A seperately trained autoencoder reduces the xyz histogram for every single of the 50 time bins, and the resulting 
(reduced_dim,t) sequence is inserted into the LSTM network, whose output is the classification task.

bs,11,13,18,50
  autoencoder
bs,seq,50
  LSTM
bs,2


"""


import numpy as np
from keras.models import Model
from keras.layers import LSTM, Input, Dropout, Dense, Flatten, Conv3D, UpSampling3D,BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D


def generate_test_input(n_bins, batchsize):
    #Dimension (batchsize,11,13,18,50,1)
    input_data=np.ones((batchsize,)+n_bins)
    return input_data     

def preprocess_batch(batch_of_histograms):
    """
    Preprocess one batch of xyzt histograms for the autoencoder training
    Input shape: (batchsize,11,13,18,50,1)
    Output shape:(batchsize*50,11,13,18,1) shuffled along first axis
    """
    batch_of_histograms=batch_of_histograms.reshape((batch_of_histograms.shape[-2]*batch_of_histograms.shape[0],)+batch_of_histograms.shape[1:-2]+(1,))

    #Shuffle along the first axis, so that the 50 spatial histograms of one event 
    #do not follow each other all the time during autoencoder training
    np.random.shuffle(batch_of_histograms)
        
    return batch_of_histograms

#Test input data
input_data = generate_test_input(n_bins=(8,8,8,20,1),batchsize=2)


input_data = preprocess_batch(input_data, shuffle=True)


def LSTM_training(input_data):
    #inputdata.shape: bs,11,13,18,50,1
    
    #train autoencoder on single time bins
    train_autoencoder_for_LSTM(preprocessed_data)
    save_autoencoder()
    
    #load the trained model
    encoder=load_encoder_for_LSTM()
    #transform 11,13,18 time bin into sequence
    LSTM_input = encoder.predict(input_batch_timebin, batch_size=32)
    
    train_LSTM(LSTM_input)
    save_LSTM()




def setup_autoencoder_test(LSTM_stage, modelpath_and_name=None):
    """
    LSTM stages:
    0:    training of autoencoder on xyz time-bins
    1:    LSTM training with frozen encoder part of autoencoder
    """
    
    trainable=False if LSTM_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    
    inputs = Input(shape=(8,8,8,1))
    x = Conv3D(filters=2, kernel_size=(3,3,3), padding="same", kernel_initializer='he_normal', trainable=trainable)(inputs)
    encoded = AveragePooling3D((4, 4, 4), padding='same')(x) #2,2,2
    
    if LSTM_stage == 0:  #The Decoder part:
        x = UpSampling3D((2, 2, 2))(encoded)
        x = Conv3DTranspose(filters=2, kernel_size=(3,3,3), padding="same", kernel_initializer='he_normal')(x)
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)

        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    else: #Replacement for the decoder part for supervised training:
        """
        #load the encoder weigths from an existing autoencoder:
        encoder = Model(inputs=inputs, outputs=encoded)
        autoencoder = load_model(modelpath_and_name)
        for i,layer in enumerate(encoder.layers):
            layer.set_weights(autoencoder.layers[i].get_weights())
        """
        
        x = Flatten()(encoded) #8,
        #Input shape for LSTM is (timesteps, data_dim)
        x = LSTM(2, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='he_normal', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)(x)
        
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model



