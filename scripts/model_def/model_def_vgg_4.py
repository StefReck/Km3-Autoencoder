# -*- coding: utf-8 -*-

from keras.models import Model, load_model
from keras.layers import Activation, Input, Dropout, Dense, Flatten, Conv3D, UpSampling3D, BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D
from keras import backend as K


#Standard Conv Blocks
def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis, strides=(1,1,1), dropout=0.0):
    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable)(inp)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
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


def setup_vgg_4_6c(autoencoder_stage, modelpath_and_name=None):
    #745,065 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
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
    
    x=conv_block(x, filters=106, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=106, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        
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
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_4_ConvAfterPool(autoencoder_stage, modelpath_and_name=None):
    # free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
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
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    encoded = conv_block(x, filters=64, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x=convT_block(encoded, filters=64, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x3x5
        x = UpSampling3D((2, 2, 2))(x) #4x6x10
        x=convT_block(x, filters=64, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=32, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        
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
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_4_6c_scale(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_base=40
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_4_8c(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_base=28
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_4_10c(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_base=26
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_4_10c_smallkernel(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=51
    kernel_size=(2,2,2)
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base, kernel_size=kernel_size, padding="same",  trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x,      filters=filter_base, kernel_size=kernel_size, padding="same",  trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x,      filters=filter_base, kernel_size=kernel_size, padding="same",  trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x,      filters=filter_base, kernel_size=kernel_size, padding="valid", trainable=train, channel_axis=channel_axis) #10x17x24
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #11,17,25
    x=conv_block(x,      filters=filter_base, kernel_size=kernel_size, padding="valid", trainable=train, channel_axis=channel_axis) #10,16,24
    x=conv_block(x,      filters=filter_base, kernel_size=kernel_size, padding="same",  trainable=train, channel_axis=channel_axis) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x,    filters=filter_base*2, kernel_size=kernel_size, padding="same",  trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x,    filters=filter_base*2, kernel_size=kernel_size, padding="same",  trainable=train, channel_axis=channel_axis) #6x8x12
    x=conv_block(x,    filters=filter_base*2, kernel_size=kernel_size, padding="valid", trainable=train, channel_axis=channel_axis) #5x7x11
    x=conv_block(x,               filters=64, kernel_size=kernel_size, padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base*2, kernel_size=kernel_size, padding="valid", channel_axis=channel_axis) #5x7x11
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #6x9x13
        x=conv_block(x, filters=filter_base*2, kernel_size=kernel_size, padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*2, kernel_size=kernel_size, padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base, kernel_size=kernel_size, padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base, kernel_size=kernel_size, padding="valid", channel_axis=channel_axis) #11x17x25
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #12x19x26
        x=conv_block(x, filters=filter_base, kernel_size=kernel_size, padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        x=convT_block(x, filters=filter_base, kernel_size=kernel_size, padding="same", channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=kernel_size, padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=kernel_size, padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=kernel_size, padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_4_10c_triple(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_base=14
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_base*4, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=filter_base*4, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=filter_base*4, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base*4, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_base*4, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*4, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_base*2, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_4_10c_triple_same_structure(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_base_enc=14
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base_enc, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base_enc, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base_enc, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_base_enc*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_base_enc*2, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x=conv_block(x, filters=filter_base_enc*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_base_enc*4, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=filter_base_enc*4, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=filter_base_enc*4, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        filter_base_dec=(50,25,12)
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base_dec[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_base_dec[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base_dec[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base_dec[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base_dec[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_base_dec[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        x=convT_block(x, filters=filter_base_dec[1], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base_dec[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base_dec[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base_dec[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_4_7c_less_filters(autoencoder_stage, modelpath_and_name=None):
    #745,065 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=26, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=26, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=26, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=26, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=52, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #4x6x10
    x=conv_block(x, filters=52, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=52, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=52, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=26, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=26, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=26, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=26, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=26, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_4_7c_same_structure(autoencoder_stage, modelpath_and_name=None):
    #745,065 free params
    #first_dec_layer: 21
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    filter_enc=32
    filter_dec=29
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_enc, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_enc, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_enc, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_enc, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_enc*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #4x6x10
    x=conv_block(x, filters=filter_enc*2, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_dec*2, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_dec*2, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_dec*2, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_dec, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_dec, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_dec, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_dec, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_4_15c(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_base=21
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base*2-1, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_base, kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base, kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_4_30c(autoencoder_stage, modelpath_and_name=None):
    #753,969 free params
    batchnorm_before_dense=True
    dropout_for_dense=0.2
    batchnorm_for_dense=False
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filter_base=[15,28]
    no_of_conv_layers=[8,10,12,   12,10,8]
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base[0], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    for i in range(no_of_conv_layers[0]-1):
        x=conv_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #10x16x24
    for i in range(no_of_conv_layers[1]-2):
        x=conv_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #11x18x25
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    for i in range(no_of_conv_layers[2]-2):
        x=conv_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="same", trainable=train, channel_axis=channel_axis) #5x8x12
    x=conv_block(x, filters=64, kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis) #4x6x10
    
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5 x 64
        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        for i in range(no_of_conv_layers[3]-3):
            x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        for i in range(no_of_conv_layers[4]-2):
            x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x25
            
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        for i in range(no_of_conv_layers[5]):
            x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

#auto = setup_vgg_4_30c(0)
#auto.summary()

"""
#find out the number by looking up the name of the first decoder layer in model.summary(),
#then get the index of that layer in model.trainable_weights[i].name
import numpy as np
def print_stats_of_trainable_weights(model, index_of_first_dec_layer):
    #first trainable layer in decoder has to have name=="first_dec"
    enc_params=0
    dec_params=0
    in_decoder=False
    for i,layer in enumerate(model.trainable_weights):
        if i == index_of_first_dec_layer:
            in_decoder=True
        if in_decoder==False:
            enc_params+=np.prod(K.get_value(layer).shape)
        else:
            dec_params+=np.prod(K.get_value(layer).shape)
    print(enc_params, dec_params, enc_params+dec_params)
print_stats_of_trainable_weights(auto, 21)
"""


