# -*- coding: utf-8 -*-

"""
Functions that return models are defined in this file
"""

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D, Cropping3D, Conv3DTranspose, AveragePooling3D


def setup_vgg_like(make_autoencoder, modelpath_and_name=None):
    #a vgg-like autoencoder, witht lots of convolutional layers
    #If make_autoencoder==False only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    inputs = Input(shape=(11,13,18,1))
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder)(inputs)
    x = Conv3D(filters=32, kernel_size=(2,2,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder)(x)
    #10x12x16 x 32
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 64
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder )(x)
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder )(x)
    x = Conv3D(filters=64, kernel_size=(2,3,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=make_autoencoder )(x)
    #4x4x6 x 64
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x2x3 x 64

    if make_autoencoder == True:
        #The Decoder part:
        
        #2x2x3 x 64
        x = UpSampling3D((2, 2, 2))(encoded)
        #4x4x6 x 64
        x = Conv3DTranspose(filters=64, kernel_size=(2,3,3), padding='valid', activation='relu' )(x)
        #5x6x8 x 64
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal' )(x)
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal' )(x)
        x = UpSampling3D((2, 2, 2))(x)
        #10x12x16 x 64
        x = Conv3DTranspose(filters=32, kernel_size=(2,2,3), padding='valid', activation='relu' )(x)
        #11x13x18 x 32
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        decoded = Conv3DTranspose(filters=1, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    elif make_autoencoder == False:
        #Replacement for the decoder part for supervised training:
        
        encoder= Model(inputs=inputs, outputs=encoded)
        encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        nb_classes=1
        outputs = Dense(nb_classes, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    
    
    
#Outdated:
def setup_conv_model_API():
    #Wie der autoencoder im sequential style, nur mit API
    
    inputs = Input(shape=(11,13,18,1))
    x = Conv3D(filters=16, kernel_size=(2,2,3), padding='valid', activation='relu')(inputs)
    #10x12x16 x 16
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 16
    x = Conv3D(filters=8, kernel_size=(3,3,3), padding='valid', activation='relu' )(x)
    #3x4x6 x 8
    encoded = Conv3D(filters=4, kernel_size=(2,3,3), padding='valid', activation='relu' )(x)
    #2x2x4 x 4
    
    
    #2x2x4 x 4
    x = Conv3DTranspose(filters=8, kernel_size=(2,3,3), padding='valid', activation='relu' )(encoded)
    #3x4x6 x 8
    x = Conv3DTranspose(filters=16, kernel_size=(3,3,3), padding='valid', activation='relu' )(x)
    #5x6x8 x 16
    x = UpSampling3D((2, 2, 2))(x)
    #10x12x16 x 16
    decoded = Conv3DTranspose(filters=1, kernel_size=(2,2,3), padding='valid', activation='relu' )(x)
    #Output 11x13x18 x 1
    
    autoencoder = Model(inputs, decoded)
    return autoencoder