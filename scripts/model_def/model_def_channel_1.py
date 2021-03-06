# -*- coding: utf-8 -*-
"""
Models for testing the channel if stuff.
"""

from keras.models import Model, load_model
from keras.layers import Activation, Reshape, TimeDistributed, Input, Dropout, Dense, Flatten, Conv3D, UpSampling3D, BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D
from keras import backend as K


#Standard Conv Blocks
def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis, strides=(1,1,1), dropout=0.0, BNunlock=False):
    #unfreeze the BN layers
    if BNunlock == True:
        BNtrainable = True
    else:
        BNtrainable = trainable
    
    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable)(inp)
    x = BatchNormalization(axis=channel_axis, trainable=BNtrainable)(x)
    x = Activation('relu', trainable=trainable)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    return x

def convT_block(inp, filters, kernel_size, padding, channel_axis, strides=(1,1,1), dropout=0.0):
    x = Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False)(inp)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    return x

def dense_block(x, units, channel_axis, batchnorm=False, dropout=0.0, activation="relu", trainable=True, name=None):
    if dropout > 0.0: 
        x = Dropout(dropout)(x)
    elif dropout < 0:
        x = Dropout(0.0)(x)#add this layer, so that loading of modelweights by going through layers works
    
    x = Dense(units=units, use_bias=1-batchnorm, kernel_initializer='he_normal', activation=None, trainable=trainable)(x)
    if batchnorm==True: x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
    if name is not None:
        x = Activation(activation, name=name)(x)
    else:
        x = Activation(activation)(x)
    return x



def setup_channel_vgg(autoencoder_stage, options_dict, modelpath_and_name=None):
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = False
    batchnorm_for_dense    = True
    number_of_filters_in_input = options_dict["number_of_filters_in_input"] #either 1 (no channel id) or 31
    
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,32,64]
    
    inputs = Input(shape=(11,13,18,number_of_filters_in_input))
    x=conv_block(inputs, filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x = AveragePooling3D((2, 2, 2), padding='same')(x) #11x18x25
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x25
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='same')(x) #5x8x12
    
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x12
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
    x = AveragePooling3D((2, 2, 2), padding='same')(x) #2x3x5

    x = Flatten()(x)
    x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
    x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
    outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def setup_channel(autoencoder_stage, options_dict, modelpath_and_name=None):
    #dropout_for_dense      = 0 #options_dict["dropout_for_dense"]
    n_bins=(11,13,18,31)
    dropout_for_conv = options_dict["dropout_for_conv"]
    neurons_in_bottleneck = options_dict["neurons_in_bottleneck"]
    model_type = options_dict["model_type"]
    # for time distributed wrappers: (batchsize, timesteps, n_bins)
    #inputs = Input(shape=(31,11,13,18))
    #x = TimeDistributed( Dense(8), input_shape=(10, 16) )(inputs)
    encoder_only_mode=options_dict["encoder_only"]
    make_stateful = options_dict["make_stateful"]
    
    if encoder_only_mode==True:
        autoencoder_stage=1
        print("Autoencoder stage set to 1 for encoder only mode")
    
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    if model_type == 0:
        units_array_enc = [256,128,128,64 ,64 ,32,32,16,8 ,neurons_in_bottleneck]
        dropout_enc =     [0  ,0.2,0.1,0.1,0.1, 0, 0, 0, 0,0]
        units_array_dec = [8,16,32,32,64,64 ,128,128,256,31]
        dropout_dec =     [0,0 , 0, 0, 0,0.1,0.1,0.1,0.2, 0]
    elif model_type==1:
        units_array_enc = [512,512,128,neurons_in_bottleneck]
        dropout_enc =     [0,0,0,0]
        units_array_dec = [128,512,512,31]
        dropout_dec =     [0,0,0,0]
    elif model_type==2:
        units_array_enc = [512,512,128,neurons_in_bottleneck]
        dropout_enc =     [0,0.1,0.1,0]
        units_array_dec = [128,512,512,31]
        dropout_dec =     [0,0.1,0.1,0]
    elif model_type==3:
        units_array_enc = [768,512,256,neurons_in_bottleneck]
        dropout_enc =     [0,  0.2,0.1,0 ]
        units_array_dec = [256,512,768,31]
        dropout_dec =     [0,  0.2,0.1,0 ]
    
    if autoencoder_stage==1 or autoencoder_stage==3:
        #no dropout if the encoder part is used frozen
        #still add dropout layer with 0 dropout though, so that load_weights works
        trainable=False #the dense layers of the channel AE
        dropout_enc=[x * (-1) for x in dropout_enc]
        dropout_dec=[x * (-1) for x in dropout_dec]
    elif autoencoder_stage==2:
        #in unfrozen stage, dense layers are unfrozen (I dont use this stage for channel networks)
        trainable=True
    
    
    if autoencoder_stage==0:
        units_array=units_array_enc+units_array_dec
        dropout_array = dropout_enc + dropout_dec
        
        inputs = Input(shape=(n_bins[-1],))
        x = dense_block(inputs, units_array[0], channel_axis, batchnorm=True, dropout=dropout_array[0])
        for i,units in enumerate(units_array[1:-1]):
            if units==neurons_in_bottleneck:
                name="encoded"
            else:
                name=None
            x = dense_block(x, units, channel_axis, batchnorm=True, dropout=dropout_array[i+1], name=name)
        outputs = dense_block(x, units_array[-1], channel_axis, batchnorm=False, dropout=dropout_array[-1], activation="linear")
        
        model = Model(inputs=inputs, outputs=outputs)
    
    else:
        inputs = Input(shape=n_bins)
        x = dense_block(inputs, units_array_enc[0], channel_axis, batchnorm=True, dropout=dropout_enc[0], trainable=trainable)
        for i,units in enumerate(units_array_enc[1:-1]):
            x = dense_block(x, units, channel_axis, batchnorm=True, dropout=dropout_enc[i+1], trainable=trainable)
        encoded = dense_block(x, units_array_enc[-1], channel_axis, batchnorm=True, dropout=dropout_enc[-1], trainable=trainable, name="encoded")
        
        
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                
                if make_stateful==True and isinstance(layer, BatchNormalization):
                    #make it so that the test mean and variance is recalculated
                    layer.stateful=True
                    print("Made layer", layer.name, "stateful.")
                    
                layer.set_weights(autoencoder.layers[i].get_weights())
            
            if encoder_only_mode==True:
                return encoder
        
        filter_base=[32,32,64]
        train=True
        unlock_BN_in_encoder=False
    
        x=conv_block(encoded, filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder, dropout=dropout_for_conv) #11x18x50
        x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder, dropout=dropout_for_conv) #11x18x50
        x = AveragePooling3D((2, 2, 2), padding='same')(x) #11x18x25
        
        x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder, dropout=dropout_for_conv) #11x18x25
        x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder, dropout=dropout_for_conv) #10x16x24
        x = AveragePooling3D((2, 2, 2), padding='same')(x) #5x8x12
        
        x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder, dropout=dropout_for_conv) #5x8x12
        x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder, dropout=dropout_for_conv) #4x6x10
        x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder, dropout=dropout_for_conv) #4x6x10
        x = AveragePooling3D((2, 2, 2), padding='same')(x) #2x3x5
    
        x = Flatten()(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=True, dropout=0.0)
        x = dense_block(x, units=16,  channel_axis=channel_axis, batchnorm=True, dropout=0.0)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
    
        
    return model


def setup_channel_tiny(autoencoder_stage, options_dict, modelpath_and_name=None):
    #dropout_for_dense      = 0 #options_dict["dropout_for_dense"]
    n_bins=(11,13,18,31)
    neurons_in_bottleneck = options_dict["neurons_in_bottleneck"]
    # for time distributed wrappers: (batchsize, timesteps, n_bins)
    #inputs = Input(shape=(31,11,13,18))
    #x = TimeDistributed( Dense(8), input_shape=(10, 16) )(inputs)
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    if autoencoder_stage==0:
        inputs = Input(shape=(n_bins[-1],))
        x = Dense(neurons_in_bottleneck)(inputs)
        outputs = Dense(31)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
    
    else:
        inputs = Input(shape=n_bins)
        encoded = Dense(inputs)(neurons_in_bottleneck, trainable=(autoencoder_stage!=1))
        
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
        
        filter_base=[32,32,64]
        train=True
        unlock_BN_in_encoder=False
    
        x=conv_block(encoded, filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
        x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
        x = AveragePooling3D((2, 2, 2), padding='same')(x) #11x18x25
        
        x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x25
        x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x24
        x = AveragePooling3D((2, 2, 2), padding='same')(x) #5x8x12
        
        x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x12
        x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
        x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
        x = AveragePooling3D((2, 2, 2), padding='same')(x) #2x3x5
    
        x = Flatten()(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=True, dropout=0.2)
        x = dense_block(x, units=16,  channel_axis=channel_axis, batchnorm=True, dropout=0.2)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
    
        
    return model

