# -*- coding: utf-8 -*-
"""
Models for testing the channel if stuff.
"""

from keras.models import Model, load_model
from keras.layers import Activation, Input, Dropout, Dense, Flatten, Conv3D, UpSampling3D, BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D
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

def dense_block(x, units, channel_axis, batchnorm=False, dropout=0.0):
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(units=units, use_bias=1-batchnorm, kernel_initializer='he_normal', activation=None)(x)
    if batchnorm==True: x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x



def setup_channel_vgg(autoencoder_stage, options_dict, modelpath_and_name=None):
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = False
    batchnorm_for_dense    = True
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,32,64]
    
    inputs = Input(shape=(11,13,18,31))
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
