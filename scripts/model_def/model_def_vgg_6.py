# -*- coding: utf-8 -*-
"""
GAN approach to autoencoders.
"""

from keras.models import Model, load_model
from keras.layers import Activation, ActivityRegularization, Cropping3D, Reshape, Input, Dropout, Dense, Flatten, Conv3D, UpSampling3D, BatchNormalization, ZeroPadding3D, Conv3DTranspose, AveragePooling3D, concatenate, TimeDistributed
from keras import backend as K
from keras.engine.topology import Layer

class ReversedGradient(Layer):
    """Layer that reverses the gradients of all variables before this layer
+    while keeping the gradients of all variables after this layer.
+
+    # Arguments
+        l: hyper-parameter lambda to control the gradient.
+
+    # Input shape
+        Arbitrary. Use the keyword argument `input_shape`
+        (tuple of integers, does not include the samples axis)
+        when using this layer as the first layer in a model.
+
+    # Output shape
        Same shape as input.
    """

    def __init__(self, l=1., **kwargs):
        super(ReversedGradient, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        if isinstance(inputs, list):
            output = [-inp + K.stop_gradient(2. * inp) for inp in inputs]
        else:
            output = -inputs + K.stop_gradient(2. * inputs)
        return output


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

def conv_block_wrap(inp, filters, kernel_size, padding, trainable, channel_axis, strides=(1,1,1), dropout=0.0, use_batchnorm=True):
    x = TimeDistributed(Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', use_bias=(not use_batchnorm), trainable=trainable))(inp)
    if use_batchnorm: x = TimeDistributed(BatchNormalization(axis=channel_axis))(x)
    x = TimeDistributed(Activation('relu', trainable=trainable))(x)
    if dropout > 0.0: x = TimeDistributed(Dropout(dropout))(x)
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

def dense_block_wrap(x, units, channel_axis, batchnorm=False, dropout=0.0, trainable=True):
    if dropout > 0.0: x = TimeDistributed(Dropout(dropout))(x)
    x = TimeDistributed(Dense(units=units, use_bias=1-batchnorm, kernel_initializer='he_normal', activation=None, trainable=trainable))(x)
    if batchnorm==True: x = TimeDistributed(BatchNormalization(axis=channel_axis, trainable=trainable))(x)
    x = TimeDistributed(Activation('relu'))(x)
        
    return x

def setup_vgg_6_200_advers(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    pretrained_autoencoder_path = options_dict["pretrained_autoencoder_path"]
    
    number_of_output_neurons=options_dict["number_of_output_neurons"]

    
    if number_of_output_neurons > 1:
        supervised_last_activation='softmax'
    else:
        supervised_last_activation='linear'
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    input_shape=(11,18,50,1)
    output_filters=1
    filter_base=[32,51,50,25]

        
    inputs = Input(shape=input_shape) #11x18x50
    x = ZeroPadding3D(((1,1),(1,1),(0,0)))(inputs) #13x20x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x48
    #x = ZeroPadding3D(((1,1),(1,1),(0,0)))(x) #13x20x48
    #x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="valid",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x46
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
        AE_out = Conv3D(filters=output_filters, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal', name="AE_output_layer")(x)
        #Output 11x13x18 x 1
        
        if pretrained_autoencoder_path != None:
            generator_model = Model(inputs=inputs, outputs=AE_out)
            pretrained_autoencoder = load_model(pretrained_autoencoder_path, compile=False) #no need to compile the model as long as only weights are read out
            print("Loading weights from model", pretrained_autoencoder_path)
            weights_loaded=0
            for i,layer in enumerate(generator_model.layers):
                layer.set_weights(pretrained_autoencoder.layers[i].get_weights())
                weights_loaded+=1
            print("Weights of ",weights_loaded, "layers were loaded.")
        
        #The adversarial part:
        #Takes the reconstruction and the original, both 11x13x18x1,
        #and concatenates them to 2x11x13x18x1, then applies Conv3D to both
        #simultaneosly
        use_batchnorm_critic=True
        #advers_in = ReversedGradient()(AE_out)
        advers_added_dim = Reshape((1,11,18,50,1))(AE_out)
        input_added_dim = Reshape((1,11,18,50,1))(inputs)
        
        x = concatenate(inputs=[advers_added_dim, input_added_dim], axis=1)
        x=conv_block_wrap(x,      filters=32, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x = TimeDistributed(AveragePooling3D((2, 2, 2), padding='same'))(x) #6x9x25
        
        x=conv_block_wrap(x,      filters=32, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x=conv_block_wrap(x,      filters=32, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x = TimeDistributed(AveragePooling3D((2, 2, 2), padding='same'))(x) #3x5x13
        
        x=conv_block_wrap(x,      filters=64, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x=conv_block_wrap(x,      filters=64, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        #2x 3x5x13 x64
        x = TimeDistributed(Flatten())(x)
        #2x 12480
        x = dense_block_wrap(x, units=256, channel_axis=channel_axis, batchnorm=False, dropout=0.1)
        x = dense_block_wrap(x, units=16, channel_axis=channel_axis, batchnorm=False, dropout=0)
        classification = TimeDistributed(Dense(2, activation="softmax", kernel_initializer='he_normal'))(x)
        #out: 2x 2
        #Target output: [ [1,0], [0,1] ]
        #                  fake,  real
        adversary = Model(inputs, classification)
        return adversary
    
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
        outputs = Dense(number_of_output_neurons, activation=supervised_last_activation, kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    

def setup_vgg_6_2000_advers(autoencoder_stage, options_dict, modelpath_and_name=None):
    batchnorm_before_dense = options_dict["batchnorm_before_dense"]
    dropout_for_dense      = options_dict["dropout_for_dense"]
    unlock_BN_in_encoder   = options_dict["unlock_BN_in_encoder"]
    batchnorm_for_dense    = options_dict["batchnorm_for_dense"]
    pretrained_autoencoder_path = options_dict["pretrained_autoencoder_path"]
    
    number_of_output_neurons=options_dict["number_of_output_neurons"]

    
    if number_of_output_neurons > 1:
        supervised_last_activation='softmax'
    else:
        supervised_last_activation='linear'
    
    train=False if autoencoder_stage == 1 else True #Freeze Encoder layers in encoder+ stage
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
 
    filter_base=[32,32,64]
    
    inputs = Input(shape=(11,18,50,1))
    x=conv_block(inputs, filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x=conv_block(x,      filters=filter_base[0], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x50
    x = AveragePooling3D((1, 1, 2), padding='valid')(x) #11x18x25
    
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #11x18x25
    x = ZeroPadding3D(((0,1),(0,0),(0,1)))(x) #12,18,26
    x=conv_block(x,      filters=filter_base[1], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #10x16x24
    x = AveragePooling3D((2, 2, 2), padding='valid')(x) #5x8x12
    
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x12
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="same",  trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #5x8x12
    x = ZeroPadding3D(((0,1),(0,0),(0,0)))(x) #6x8x12
    x=conv_block(x,      filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=train, channel_axis=channel_axis, BNunlock=unlock_BN_in_encoder) #4x6x10
    encoded = AveragePooling3D((2, 2, 2), padding='valid')(x) #2x3x5

    if autoencoder_stage == 0:  #The Decoder part:
        #2x2x3

        x = UpSampling3D((2, 2, 2))(encoded) #4x6x10
        x=convT_block(x, filters=filter_base[2], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #6x8x12
        x = ZeroPadding3D(((0,1),(1,1),(1,1)))(x) #7x10x14
        x=conv_block(x,  filters=filter_base[2], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #5x8x12
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #5x8x12
        
        x = UpSampling3D((2, 2, 2))(x) #10x16x24
        x=convT_block(x, filters=filter_base[1], kernel_size=(3,3,3), padding="valid", channel_axis=channel_axis) #12x18x26
        x = ZeroPadding3D(((0,1),(1,1),(0,1)))(x) #13,20,27
        x=conv_block(x,  filters=filter_base[0], kernel_size=(3,3,3), padding="valid", trainable=True, channel_axis=channel_axis) #11x18x25
        
        x = UpSampling3D((1, 1, 2))(x) #11x18x50
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        x=convT_block(x, filters=filter_base[0], kernel_size=(3,3,3), padding="same", channel_axis=channel_axis) #11x18x50
        
        AE_out = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', activation='linear', kernel_initializer='he_normal')(x)
        #Output 11x13x18 x 1
        
        if pretrained_autoencoder_path != None:
            generator_model = Model(inputs=inputs, outputs=AE_out)
            #models/vgg_3_eps/trained_vgg_3_eps_autoencoder_epoch119.h5
            pretrained_autoencoder = load_model(pretrained_autoencoder_path, compile=False) #no need to compile the model as long as only weights are read out
            print("Loading weights from model", pretrained_autoencoder_path)
            weights_loaded=0
            for i,layer in enumerate(generator_model.layers):
                layer.set_weights(pretrained_autoencoder.layers[i].get_weights())
                weights_loaded+=1
            print("Weights of ",weights_loaded, "layers were loaded.")
        
        #The adversarial part:
        #Takes the reconstruction and the original, both 11x13x18x1,
        #and concatenates them to 2x11x13x18x1, then applies Conv3D to both
        #simultaneosly
        use_batchnorm_critic=True
        #advers_in = ReversedGradient()(AE_out)
        advers_added_dim = Reshape((1,11,18,50,1))(AE_out)
        input_added_dim = Reshape((1,11,18,50,1))(inputs)
        
        x = concatenate(inputs=[advers_added_dim, input_added_dim], axis=1)
        x=conv_block_wrap(x,      filters=32, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x = TimeDistributed(AveragePooling3D((2, 2, 2), padding='same'))(x) #6x9x25
        
        x=conv_block_wrap(x,      filters=32, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x=conv_block_wrap(x,      filters=32, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x = TimeDistributed(AveragePooling3D((2, 2, 2), padding='same'))(x) #3x5x13
        
        x=conv_block_wrap(x,      filters=64, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        x=conv_block_wrap(x,      filters=64, kernel_size=(3,3,3), padding="same",  trainable=True, channel_axis=channel_axis, dropout=0.1, use_batchnorm=use_batchnorm_critic)
        #2x 3x5x13 x64
        x = TimeDistributed(Flatten())(x)
        #2x 12480
        x = dense_block_wrap(x, units=256, channel_axis=channel_axis, batchnorm=False, dropout=0.1)
        x = dense_block_wrap(x, units=16, channel_axis=channel_axis, batchnorm=False, dropout=0)
        
        """
        #Softmax  Categorical output
        classification = TimeDistributed(Dense(2, activation="softmax", kernel_initializer='he_normal'))(x)
        #out: 2x 2
        #Target output: [ [1,0], [0,1] ]
        #                  fake,  real
        """
        #For wasserstein distance: 
        classification = TimeDistributed(Dense(1, activation="linear", kernel_initializer='he_normal'))(x)
        #out: 2x 2
        #Target output: [ -1, 1 ]
        #                  fake,  real
        
        adversary = Model(inputs, classification)
        return adversary
    
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
        outputs = Dense(number_of_output_neurons, activation=supervised_last_activation, kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model