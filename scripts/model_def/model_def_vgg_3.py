# -*- coding: utf-8 -*-
"""
Contains Definitions of setup_vgg_3, setup_vgg_3_max, setup_vgg_3_stride, setup_vgg_3_dropout

Vgg-like autoencoder-networks with 7+7 convolutional layers w/ batch norm; ca 720k free params
Just like vvg_2, but only size 3 Kernels this time
Input Format: 11x18x50 (XZT DATA)

autoencoder_stage: Type of training/network
    0: autoencoder
    1: encoder+ from autoencoder w/ frozen layers
    2: encoder+ from scratch, completely unfrozen
    
If autoencoder_stage==1 only the first part of the autoencoder (encoder part) will be generated
These layers are frozen then.
The weights of the original model can be imported then by using load_weights('xxx.h5', by_name=True)

modelpath_and_name is used to load the encoder part for supervised training, 
and only needed if make_autoencoder==False
    
"""
from keras.models import Model
from keras.layers import Activation, Input, Dropout, Dense, Flatten, Conv3D, MaxPooling3D, UpSampling3D,BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D, Reshape
from keras.layers import Lambda
from keras import backend as K
from keras import regularizers

from util.custom_layers import MaxUnpooling3D

#Standard Conv Blocks
def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis, strides=(1,1,1), dropout=0.0, ac_reg_penalty=0):
    regular = regularizers.l2(ac_reg_penalty) if ac_reg_penalty is not 0 else None
    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable, activity_regularizer=regular)(inp)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
    x = Activation('relu', trainable=trainable)(x)
    if dropout > 0.0: x = Dropout(dropout)(x)
    return x

def convT_block(inp, filters, kernel_size, padding, channel_axis, strides=(1,1,1), dropout=0.0, ac_reg_penalty=0):
    regular = regularizers.l2(ac_reg_penalty) if ac_reg_penalty is not 0 else None
    x = Conv3DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, activity_regularizer=regular)(inp)
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

def zero_center_and_normalize(x):
    x-=K.mean(x, axis=1, keepdims=True)
    x=x/K.std(x, axis=1, keepdims=True)
    return x

def setup_vgg_3(autoencoder_stage, modelpath_and_name=None):
    #832k params
    normalize_before_dense=False
    batchnorm_before_dense=False
    dropout_for_dense=0.0
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
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
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
            
        x = Flatten()(encoded)
        if normalize_before_dense==True: x = Lambda( zero_center_and_normalize )(x)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        #x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        #x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
def setup_vgg_3_reg(autoencoder_stage, modelpath_and_name=None):
    #832k params with an activity l2 penalty
    ac_reg_penalty = 1e-8 if autoencoder_stage is not 1 else 0
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #11x18x50
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis, ac_reg_penalty=ac_reg_penalty) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
def setup_vgg_3_small(autoencoder_stage, modelpath_and_name=None):
    #832k params + 983k dense
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
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
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5 x 64
    x = Flatten()(x)#1920
    encoded = Dense(256, activation='relu', kernel_initializer='he_normal', trainable=train)(x)
    
    if autoencoder_stage == 0:  #The Decoder part:
        x = Dense(1920, activation='relu', kernel_initializer='he_normal')(encoded)
        x = Reshape( (2,3,5,64) )(x)
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(x) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Dense(1024, activation='relu', kernel_initializer='he_normal')(encoded)
        x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_3_verysmall(autoencoder_stage, modelpath_and_name=None):
    #832k params + na k dense
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
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
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5 x 64
    x = Flatten()(x)#1920
    x = Dense(256, activation='relu', kernel_initializer='he_normal', trainable=train)(x)
    encoded = Dense(16, activation='relu', kernel_initializer='he_normal', trainable=train)(x)
    
    if autoencoder_stage == 0:  #The Decoder part:
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(encoded)
        x = Dense(1920, activation='relu', kernel_initializer='he_normal')(x)
        x = Reshape( (2,3,5,64) )(x)
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(x) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Dense(1024, activation='relu', kernel_initializer='he_normal')(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


    
def setup_vgg_3_dropout(autoencoder_stage, modelpath_and_name=None):
    #832k params
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    dropout_rate=0.1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", dropout=dropout_rate, trainable=train, channel_axis=channel_axis) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", dropout=dropout_rate, trainable=train, channel_axis=channel_axis) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", dropout=dropout_rate, channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", dropout=dropout_rate, trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", dropout=dropout_rate, channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", dropout=dropout_rate, trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", dropout=dropout_rate, channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_3_max(autoencoder_stage, modelpath_and_name=None):
    #713k params
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = MaxPooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x = MaxPooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    encoded = MaxPooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = MaxUnpooling3D(encoded) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = MaxUnpooling3D(x) #10x16x24
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = MaxUnpooling3D(x, Kernel_size=(1,1,2)) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_3_stride(autoencoder_stage, modelpath_and_name=None):
    #832k params
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", strides=(1,1,2), trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", strides=(2,2,2), trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    encoded=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", strides=(2,2,2), trainable=train, channel_axis=channel_axis) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x=convT_block(encoded, filters=64, kernel_size=(3,3,3), padding="valid", strides=(2,2,2), channel_axis=channel_axis) #5x7x11
        x = ZeroPadding3D(((0,0),(0,1),(0,1)))(x) #5x8x12
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", strides=(2,2,2), channel_axis=channel_axis) #11x17x25
        x = ZeroPadding3D(((0,0),(0,1),(0,0)))(x) #11,18,25
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", strides=(1,1,2), channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
def setup_vgg_3_stride_noRelu(autoencoder_stage, modelpath_and_name=None):
    #832k params
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="same", strides=(1,1,2), trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", strides=(2,2,2), trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    
    encoded=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", strides=(2,2,2), trainable=train, channel_axis=channel_axis) #2x3x5
    
    x = Conv3D(filters=64, kernel_size=(3,3,3), strides=(2,2,2), padding="valid", kernel_initializer='he_normal', use_bias=False, trainable=train)(x)
    encoded = BatchNormalization(axis=channel_axis, trainable=train)(x)
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x=convT_block(encoded, filters=64, kernel_size=(3,3,3), padding="valid", strides=(2,2,2), channel_axis=channel_axis) #5x7x11
        x = ZeroPadding3D(((0,0),(0,1),(0,1)))(x) #5x8x12
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", strides=(2,2,2), channel_axis=channel_axis) #11x17x25
        x = ZeroPadding3D(((0,0),(0,1),(0,0)))(x) #11,18,25
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", strides=(1,1,2), channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_3_duo(autoencoder_stage, modelpath_and_name=None):
    #With 2 additional Conv layer in the frozen encoder part compared to vgg_3
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
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
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=32, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder= Model(inputs=inputs, outputs=encoded)
            encoder.load_weights(modelpath_and_name, by_name=True)
        x = Flatten()(encoded)
        x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = Dense(16, activation='relu', kernel_initializer='he_normal')(x)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
