# -*- coding: utf-8 -*-

"""
Bottleneck study
"""

from keras.models import Model, load_model
from keras.layers import Activation, ActivityRegularization, Cropping3D, Reshape, Input, Dropout, Dense, Flatten, Conv3D, UpSampling3D, BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D
from keras import backend as K

#Standard Conv Blocks
def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis, strides=(1,1,1), dropout=0.0, BNunlock=False, name_of_first_layer = None):
    #unfreeze the BN layers
    if BNunlock == True:
        BNtrainable = True
    else:
        BNtrainable = trainable
    
    if name_of_first_layer == None:
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable)(inp)
    else:
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable, name = name_of_first_layer)(inp)
    
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

def dense_block(x, units, channel_axis, batchnorm=False, dropout=0.0, trainable=True, name=None):
    if dropout > 0.0: x = Dropout(dropout)(x)
    x = Dense(units=units, use_bias=1-batchnorm, kernel_initializer='he_normal', activation=None, trainable=trainable)(x)
    if batchnorm==True: x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
    
    if name is not None:
        x = Activation('relu', name=name)(x)
    else:
        x = Activation('relu')(x)
        
    return x




def setup_vgg_5_picture(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    encoded_penalty        = options_dict["encoded_penalty"]
    additional_conv_layer_for_encoder = options_dict["add_conv_layer"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,39,50,50]
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    x = ZeroPadding3D(((0,0),(0,1),(0,1)))(x) #2x4x6
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x6
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x6
    encoded = AveragePooling3D((1, 2, 2), padding='valid')(x) #2x2x3 x 50

    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3
        if encoded_penalty == 0:
            x = UpSampling3D((1, 2, 2))(encoded) #2x4x6
        else:
            x = ActivityRegularization(l1=encoded_penalty, l2=0.0)(encoded)
            x = UpSampling3D((1, 2, 2))(x) #2x4x6
        
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x4x6
        x = ZeroPadding3D(((1,1),(0,1),(0,1)))(x) #4x5x7 
        x=conv_block(x,  filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #2x3x5

        x = UpSampling3D((2, 2, 2))(x) #4x6x10
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name, compile=False)
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
        #2x2x3 x50
        if additional_conv_layer_for_encoder == True:
             x = conv_block(encoded, filters=filter_base[3], kernel_size=(2,2,2), padding="same",  trainable=True, channel_axis=channel_axis, name_of_first_layer = "after_encoded")
             x = Flatten()(x)
        else:
            x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_5_channel(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,60,50,20]
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x3x5
    encoded=conv_block(x,filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x3x5
 
    if autoencoder_stage == 0:  #The Decoder part:
        #2x3x5
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x3x5
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x3x5
        
        x = UpSampling3D((2, 2, 2))(x) #4x6x10
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
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
        units=[100,16] #256,16
        x = dense_block(x, units=units[0], channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=units[1], channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_5_morefilter(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[28,29,40,75]
    
    inputs = Input(shape=(11,18,50,1))
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x48
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x48
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x46
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x23
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x23
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,24
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x22
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x11
    
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(x) #7x10x11
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x9
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #6x8x10
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x8
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x4
    
    x = ZeroPadding3D(((0,0),(0,1),(0,0)))(x) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    encoded = AveragePooling3D((1, 2, 2), padding='valid')(x) #2x2x2
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3
        x = UpSampling3D((1, 2, 2))(encoded) #2x4x4
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x4x4
        x = ZeroPadding3D(((1,1),(0,1),(1,1)))(x) #4x5x6
        x=conv_block(x,  filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #2x3x4

        x = UpSampling3D((2, 2, 2))(x) #4x6x8
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x10
        x = ZeroPadding3D(((0,1),(1,1),(1,2)))(x) #7x10x13
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x11
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x22
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x24
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,25
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x23
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x46
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x48
        x=Cropping3D(((1,1),(1,1),(0,0)))(x) #11x18x48
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x50
        x=Cropping3D(((1,1),(1,1),(0,0)))(x)#11x18x50
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


def setup_vgg_5_200(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    
    filter_base_version    = options_dict["filter_base_version"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    if filter_base_version == "standard":
        filter_base=[32,51,50,25]
    elif filter_base_version == "large":
        filter_base=[64,75,75,25]
        
    inputs = Input(shape=(11,18,50,1))
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x48
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x48
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x46
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x23
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x23
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,24
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x22
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x11
    
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(x) #7x10x11
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x9
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #6x8x10
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x8
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x4
    
    x = ZeroPadding3D(((0,0),(0,1),(0,0)))(x) #2x4x4
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    encoded = AveragePooling3D((1, 2, 2), padding='valid')(x) #2x2x2
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3
        x = UpSampling3D((1, 2, 2))(encoded) #2x4x4
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x4x4
        x = ZeroPadding3D(((1,1),(0,1),(1,1)))(x) #4x5x6
        x=conv_block(x,  filters=filter_base[3], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #2x3x4
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x4x4
        
        x = UpSampling3D((2, 2, 2))(x) #4x6x8
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x10
        x = ZeroPadding3D(((0,1),(1,1),(1,2)))(x) #7x10x13
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x11
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x22
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x24
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,25
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x23
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x46
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x48
        x=Cropping3D(((1,1),(1,1),(0,0)))(x) #11x18x48
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x50
        x=Cropping3D(((1,1),(1,1),(0,0)))(x)#11x18x50
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name, compile=False) #no need to compile the model as long as only weights are read out
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup_vgg_5_200_deep(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,51,50,25]
        
    inputs = Input(shape=(11,18,50,1))
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x48
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x48
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x46
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder)
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x23
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x23
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,24
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x22
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder)
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder)
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x11
    
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(x) #7x10x11
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x9
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #6x8x10
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x8
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder)
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder)
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x4
    
    x = ZeroPadding3D(((0,0),(0,1),(0,0)))(x) #2x4x4
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) 
    encoded = AveragePooling3D((1, 2, 2), padding='valid')(x) #2x2x2
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3
        x = UpSampling3D((1, 2, 2))(encoded) #2x4x4
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x4x4
        x = ZeroPadding3D(((1,1),(0,1),(1,1)))(x) #4x5x6
        x=conv_block(x,  filters=filter_base[3], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #2x3x4
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x4x4
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        
        x = UpSampling3D((2, 2, 2))(x) #4x6x8
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x10
        x = ZeroPadding3D(((0,1),(1,1),(1,2)))(x) #7x10x13
        x=conv_block(x,  filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x11
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x22
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x24
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,25
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x23
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis)
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x46
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x48
        x=Cropping3D(((1,1),(1,1),(0,0)))(x) #11x18x48
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x50
        x=Cropping3D(((1,1),(1,1),(0,0)))(x)#11x18x50
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) 
        decoded = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        autoencoder = Model(inputs, decoded)
        return autoencoder
    else: #Replacement for the decoder part for supervised training:
        if autoencoder_stage == 1: #Load weights of encoder part from existing autoencoder
            encoder = Model(inputs=inputs, outputs=encoded)
            autoencoder = load_model(modelpath_and_name, compile=False) #no need to compile the model as long as only weights are read out
            for i,layer in enumerate(encoder.layers):
                layer.set_weights(autoencoder.layers[i].get_weights())
            
        x = Flatten()(encoded)
        if batchnorm_before_dense==True: x = BatchNormalization(axis=channel_axis)(x)
        x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_5_200_dense(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,32,42,42]
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5
    
    x = ZeroPadding3D(((0,0),(0,1),(0,1)))(x) #2x4x6
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x6
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x6
    x = AveragePooling3D((1, 2, 2), padding='valid')(x) #2x2x3
    x = Flatten()(x)
    encoded = dense_block(x, units=200, channel_axis=channel_axis, batchnorm=True, dropout=0, trainable=train, name="encoded")
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3
        x = dense_block(encoded, units=2*2*3*filter_base[3], channel_axis=channel_axis, batchnorm=True, dropout=0)
        x = Reshape((2,2,3,filter_base[3]))(x)
        x = UpSampling3D((1, 2, 2))(x) #2x4x6
        x=convT_block(x, filters=filter_base[3], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x4x6
        x = ZeroPadding3D(((1,1),(0,1),(0,1)))(x) #4x5x7 
        x=conv_block(x,  filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #2x3x5

        x = UpSampling3D((2, 2, 2))(x) #4x6x10
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
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
            
        #x = Flatten()(encoded)
        if batchnorm_before_dense==True: 
            x = BatchNormalization(axis=channel_axis)(encoded)
            x = dense_block(x, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        else:
            x = dense_block(encoded, units=256, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        x = dense_block(x, units=16, channel_axis=channel_axis, batchnorm=batchnorm_for_dense, dropout=dropout_for_dense)
        outputs = Dense(2, activation='softmax', kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model


def setup_vgg_5_64(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,40,44,46,64]
    
    inputs = Input(shape=(11,18,50,1))
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x48
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x48
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x46
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x23
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x23
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,24
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x22
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x11
    
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(x) #7x10x11
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x9
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #6x8x10
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x8
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x4
    
    x = ZeroPadding3D(((0,0),(0,1),(0,0)))(x) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x = AveragePooling3D((1, 2, 2), padding='valid')(x) #2x2x2
    
    x=conv_block(x,      filters=filter_base[4], kernel_size=(2,2,2), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x2x2
    x=conv_block(x,      filters=filter_base[4], kernel_size=(2,2,2), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x2x2
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #1x1x1
    
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3
        x = UpSampling3D((2, 2, 2))(encoded) #2x2x2
        x=convT_block(x, filters=filter_base[4], kernel_size=(2,2,2), padding="same", channel_axis=channel_axis) #2x4x4
        x=convT_block(x, filters=filter_base[3], kernel_size=(2,2,2), padding="same", channel_axis=channel_axis) #2x4x4
        
        x = UpSampling3D((1, 2, 2))(x) #2x4x4
        x = ZeroPadding3D(((1,1),(0,1),(1,1)))(x) #4x5x6
        x=conv_block(x,  filters=filter_base[3], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #2x3x4
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x3x4
        
        x = UpSampling3D((2, 2, 2))(x) #4x6x8
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x10
        x = ZeroPadding3D(((0,1),(1,1),(1,2)))(x) #7x10x13
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x11
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x22
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x24
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,25
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x23
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x46
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x48
        x=Cropping3D(((1,1),(1,1),(0,0)))(x) #11x18x48
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x50
        x=Cropping3D(((1,1),(1,1),(0,0)))(x)#11x18x50
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


def setup_vgg_5_32(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    filter_base=[32,32,60,44,32]
    
    inputs = Input(shape=(11,18,50,1))
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x48
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x48
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x46
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x23
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x23
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,24
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x22
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x11
    
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(x) #7x10x11
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x9
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #6x8x10
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x8
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x4
    
    x = ZeroPadding3D(((0,0),(0,1),(0,0)))(x) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x=conv_block(x,      filters=filter_base[3], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x4x4
    x = AveragePooling3D((1, 2, 2), padding='valid')(x) #2x2x2
    
    x=conv_block(x,      filters=filter_base[4], kernel_size=(2,2,2), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x2x2
    x=conv_block(x,      filters=filter_base[4], kernel_size=(2,2,2), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #2x2x2
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #1x1x1
    
    
    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3
        x = UpSampling3D((2, 2, 2))(encoded) #2x2x2
        x=convT_block(x, filters=filter_base[4], kernel_size=(2,2,2), padding="same", channel_axis=channel_axis) #2x4x4
        x=convT_block(x, filters=filter_base[3], kernel_size=(2,2,2), padding="same", channel_axis=channel_axis) #2x4x4
        
        x = UpSampling3D((1, 2, 2))(x) #2x4x4
        x = ZeroPadding3D(((1,1),(0,1),(1,1)))(x) #4x5x6
        x=conv_block(x,  filters=filter_base[3], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #2x3x4
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #2x3x4
        
        x = UpSampling3D((2, 2, 2))(x) #4x6x8
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x10
        x = ZeroPadding3D(((0,1),(1,1),(1,2)))(x) #7x10x13
        x=conv_block(x,  filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x11
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x22
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x24
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,25
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x23
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x46
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x48
        x=Cropping3D(((1,1),(1,1),(0,0)))(x) #11x18x48
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #13x20x50
        x=Cropping3D(((1,1),(1,1),(0,0)))(x)#11x18x50
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


#setup_vgg_5_morefilter(0, 0).summary()
#setup_vgg_5_channel(0).summary()


