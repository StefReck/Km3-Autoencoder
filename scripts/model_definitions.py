# -*- coding: utf-8 -*-

"""
Functions that return models are defined in this file
"""

from keras.models import Model
from keras.layers import Activation, Input, Dense, Flatten, Conv3D, MaxPooling3D, UpSampling3D,BatchNormalization, ZeroPadding3D, Cropping3D, Conv3DTranspose, Reshape, AveragePooling3D
from keras import backend as K

from util.custom_layers import MaxUnpooling3D
from model_def.model_def_vgg_2_xzt import setup_vgg_2,setup_vgg_2_dropout, setup_vgg_2_max, setup_vgg_2_stride

#Standard Conv Blocks
def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis):
    x = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable)(inp)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
    out = Activation('relu', trainable=trainable)(x)
    return out

def convT_block(inp, filters, kernel_size, padding, channel_axis):
    x = Conv3DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal', use_bias=False)(inp)
    x = BatchNormalization(axis=channel_axis)(x)
    out = Activation('relu')(x)
    return out

def setup_vgg_like(autoencoder_stage, modelpath_and_name=None):
    #a vgg-like autoencoder, witht lots of convolutional layers
    #tag: vgg_0
    
    #autoencoder_stage: Type of training/network
    # 0: autoencoder
    # 1: encoder+ from autoencoder w/ frozen layers
    # 2: encoder+ from scratch, completely unfrozen
    
    #If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    if autoencoder_stage == 1:
        #Freeze encoder layers
        train=False
    else:
        train=True
    
    inputs = Input(shape=(11,13,18,1))
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=train)(inputs)
    x = Conv3D(filters=32, kernel_size=(2,2,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=train)(x)
    #10x12x16 x 32
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 32
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=train )(x)
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation='relu', kernel_initializer='he_normal', trainable=train )(x)
    x = Conv3D(filters=64, kernel_size=(2,3,3), padding='valid', activation='relu', kernel_initializer='he_normal', trainable=train )(x)
    #4x4x6 x 64
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x2x3 x 64

    if autoencoder_stage == 0:
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
    
    else:
        #Replacement for the decoder part for supervised training:
        
        if autoencoder_stage == 1:
            #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    
    
def setup_vgg_1(autoencoder_stage, modelpath_and_name=None):
    #enhanced version of vgg_0, with zero_center compatibility and batch normalization
    #tag: vgg_1
    
    #autoencoder_stage: Type of training/network
    # 0: autoencoder
    # 1: encoder+ from autoencoder w/ frozen layers
    # 2: encoder+ from scratch, completely unfrozen
    
    #If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    if autoencoder_stage == 1:
        #Freeze encoder layers
        train=False
    else:
        train=True
        
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,13,18,1))
    
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(inputs)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=32, kernel_size=(2,2,3), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #10x12x16 x 32
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 32
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=64, kernel_size=(2,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #4x4x6 x 64
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x2x3 x 64

    if autoencoder_stage == 0:
        #The Decoder part:
        
        #2x2x3 x 64
        x = UpSampling3D((2, 2, 2))(encoded)
        #4x4x6 x 64
        
        x = Conv3DTranspose(filters=64, kernel_size=(2,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        #5x6x8 x 64
        
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
    
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
         
        x = UpSampling3D((2, 2, 2))(x)
        #10x12x16 x 64
        
        x = Conv3DTranspose(filters=32, kernel_size=(2,2,3), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #11x13x18 x 32
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    else:
        #Replacement for the decoder part for supervised training:
        
        if autoencoder_stage == 1:
            #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    

def setup_vgg_1_xzt(autoencoder_stage, modelpath_and_name=None):
    #enhanced version of vgg_0, with zero_center compatibility and batch normalization
    #format 11x18x50
    #713k params
    
    #autoencoder_stage: Type of training/network
    # 0: autoencoder
    # 1: encoder+ from autoencoder w/ frozen layers
    # 2: encoder+ from scratch, completely unfrozen
    
    #If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    if autoencoder_stage == 1:
        #Freeze encoder layers
        train=False
    else:
        train=True
        
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(inputs)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x)
    #11x18x25
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=32, kernel_size=(2,3,2), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x8x12
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=64, kernel_size=(2,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #2x3x5

    if autoencoder_stage == 0:
        #The Decoder part:
        
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded)
        #4x6x10
        
        x = Conv3DTranspose(filters=64, kernel_size=(2,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #5x8x12
        
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
    
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
         
        x = UpSampling3D((2, 2, 2))(x)
        #10x16x24
        
        x = Conv3DTranspose(filters=32, kernel_size=(2,3,2), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #11x18x25
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        x = UpSampling3D((1, 1, 2))(x)
        #11x18x50
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    else:
        #Replacement for the decoder part for supervised training:
        
        if autoencoder_stage == 1:
            #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    

def setup_vgg_1_xzt_max(autoencoder_stage, modelpath_and_name=None):
    #like vgg_1_xzt but with Max/Unmaxpooling
    #format 11x18x50    (=9900)
    
    #autoencoder_stage: Type of training/network
    # 0: autoencoder
    # 1: encoder+ from autoencoder w/ frozen layers
    # 2: encoder+ from scratch, completely unfrozen
    
    #If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    if autoencoder_stage == 1:
        #Freeze encoder layers
        train=False
    else:
        train=True
        
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    print("Loading model vgg_1_xzt_max")
    inputs = Input(shape=(11,18,50,1))
    
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(inputs)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #11x18x50
    x = MaxPooling3D((1, 1, 2), padding='valid')(x)
    #11x18x25
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=32, kernel_size=(2,3,2), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #10x16x24
    x = MaxPooling3D((2, 2, 2), padding='valid')(x)
    #5x8x12
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=64, kernel_size=(2,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #4x6x10
    encoded = MaxPooling3D((2, 2, 2), padding='valid')(x)
    #2x3x5 x 64    (=1920 = 19.4 % org size)

    if autoencoder_stage == 0:
        #The Decoder part:
        print("Loading Decoder")
        #2x3x5 x 64
        x = MaxUnpooling3D(encoded)
        #4x6x10
        
        x = Conv3DTranspose(filters=64, kernel_size=(2,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #5x8x12
        
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
    
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
         
        x = MaxUnpooling3D(x)
        #10x16x24
        
        x = Conv3DTranspose(filters=32, kernel_size=(2,3,2), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #11x18x25
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        x = MaxUnpooling3D(x,(1,1,2))
        #11x18x50
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    else:
        #Replacement for the decoder part for supervised training:
        print("Loading dense")
        if autoencoder_stage == 1:
            #Load weights of encoder part from existing autoencoder
            print("Loading weights of existing autoencoder", modelpath_and_name)
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    
def setup_vgg_1_xzt_stride(autoencoder_stage, modelpath_and_name=None):
    #like vgg1xzt, but with stride>1 instead of pooling
    #format 11x18x50
    #750k params
    
    #autoencoder_stage: Type of training/network
    # 0: autoencoder
    # 1: encoder+ from autoencoder w/ frozen layers
    # 2: encoder+ from scratch, completely unfrozen
    
    #If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
    #These layers are frozen then
    #The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)
    
    #modelpath_and_name is used to load the encoder part for supervised training, 
    #and only needed if make_autoencoder==False
    
    if autoencoder_stage == 1:
        #Freeze encoder layers
        train=False
    else:
        train=True
        
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(inputs)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    
    #11x18x50
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', strides=(1,1,2), kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    #11x18x25
    
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x)
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    #10x16x24
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', strides=(2,2,2), kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    #5x8x12
    
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x)
    #6x8x12
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    #4x6x10
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', strides=(2,2,2), kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    x = BatchNormalization(axis=channel_axis, trainable=train)(x)
    encoded = Activation('relu', trainable=train)(x)
    #2x3x5


    if autoencoder_stage == 0:
        #The Decoder part:
        
        #2x3x5 x 64
        
        x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), padding='same', strides=(2,2,2), kernel_initializer='he_normal', use_bias=False)(encoded)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #4x6x10
        
        x = ZeroPadding3D(((1,2),(2,2),(2,2)))(x)
        #7x10x14
        x = Conv3D(filters=64, kernel_size=(3,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #5x8x12
    
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', strides=(2,2,2), kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #10x16x24
        
        x = ZeroPadding3D(((1,2),(2,2),(1,2)))(x)
        #13x20x27
        x = Conv3D(filters=32, kernel_size=(3,3,3), padding='valid', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #11x18x25
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', strides=(1,1,2), kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        #11x18x50
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    
    else:
        #Replacement for the decoder part for supervised training:
        
        if autoencoder_stage == 1:
            #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    
    
def setup_model(model_tag, autoencoder_stage, modelpath_and_name=None):
    if model_tag == "vgg_0":
        model = setup_vgg_like(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_1":
        model = setup_vgg_1(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_1_xzt":
        model = setup_vgg_1_xzt(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_1_xzt_max":
        model = setup_vgg_1_xzt_max(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_1_xzt_stride":
        model = setup_vgg_1_xzt_stride(autoencoder_stage, modelpath_and_name)
        
    elif model_tag == "vgg_2":
        model = setup_vgg_2(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_2_dropout":
        model = setup_vgg_2_dropout(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_2_max":
        model = setup_vgg_2_max(autoencoder_stage, modelpath_and_name)
    elif model_tag == "vgg_2_stride":
        model = setup_vgg_2_stride(autoencoder_stage, modelpath_and_name)
    
    else:
        raise Exception('Model tag not available: '+ model_tag)
    return model

    
#For testing purposes
#model = setup_model("vgg_1_xzt_stride", 0)
#model2 = setup_model("vgg_2_dropout", 2)
#model3 = setup_model("vgg_1_xzt_stride", 0)
#model.compile(optimizer=adam, loss='mse')

"""
#from numpy import prod
def summ_model(model):
    shape=0
    deep=0
    print("Depth\tSize\tShape")
    for depth,layer in enumerate(model.layers):
        deep=depth
        #shape=layer.output_shape[1:]
        #print(depth, "\t", prod(shape),"\t", shape)
    print(deep)
"""

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
