# -*- coding: utf-8 -*-

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Flatten, BatchNormalization, Activation, Conv3D, MaxPooling3D, UpSampling3D, Conv2D, Conv2DTranspose, ZeroPadding3D, Cropping3D, Conv3DTranspose, AveragePooling3D
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import h5py
from keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages

from compare_hists import *
from util.Loggers import *


def setup_simple_model():
    global model
    model=Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    
def setup_conv_model():
    #Test autoencoder im sequential style
    global conv_model
    conv_model = Sequential()
    #INput: 11x13x18 x 1
    
    conv_model.add(Conv3D(filters=16, kernel_size=(2,2,3), padding='valid', activation='relu', input_shape=(11, 13, 18, 1)))
    #10x12x16 x 16
    conv_model.add(AveragePooling3D((2, 2, 2), padding='valid'))
    #5x6x8 x 16
    conv_model.add(Conv3D(filters=8, kernel_size=(3,3,3), padding='valid', activation='relu' ))
    #3x4x6 x 8
    conv_model.add(Conv3D(filters=4, kernel_size=(2,3,3), padding='valid', activation='relu' ))
    #2x2x4 x 4

    #2x2x4 x 4
    conv_model.add(Conv3DTranspose(filters=8, kernel_size=(2,3,3), padding='valid', activation='relu' ))
    #3x4x6 x 8
    conv_model.add(Conv3DTranspose(filters=16, kernel_size=(3,3,3), padding='valid', activation='relu' ))
    #5x6x8 x 16
    conv_model.add(UpSampling3D((2, 2, 2)))
    #10x12x16 x 16
    conv_model.add(Conv3DTranspose(filters=1, kernel_size=(2,2,3), padding='valid', activation='relu' ))
    
    #Output 11x13x18 x 1
    conv_model.compile(optimizer='adadelta', loss='mse')
    
    #conv_model.summary()

def setup_conv_model_API():
    #Wie der autoencoder im sequential style, nur mit API
    global autoencoder
    
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
    autoencoder.compile(optimizer='adadelta', loss='mse')

def encoded_conv_model_API():
    #Erste Hälfte des conv_model_API autoencoder, um zu testen, wie import funktioniert.
    
    inputs = Input(shape=(11,13,18,1))
    x = Conv3D(filters=16, kernel_size=(2,2,3), padding='valid', activation='relu', trainable=False)(inputs)
    #10x12x16 x 16
    x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    #5x6x8 x 16
    x = Conv3D(filters=8, kernel_size=(3,3,3), padding='valid', activation='relu', trainable=False )(x)
    #3x4x6 x 8
    encoded = Conv3D(filters=4, kernel_size=(2,3,3), padding='valid', activation='relu', trainable=False )(x)
    #2x2x4 x 4

    autoencoder = Model(inputs, encoded)
    return autoencoder
    
def test_model():
    inputs = Input(shape=(5,5,1))
    x = Conv2D(filters=1, strides=1, kernel_size=(3,3), padding='same', activation='relu')(inputs)
    #2,2
    x = Conv2DTranspose(filters=1, strides=1, kernel_size=(3,3), padding='same', activation='relu')(x)
    #12,14,18

    #stride 1/valid             stride 1/same = 1pad --> 13,15,20
    #11,13,18                   
    #9,11,16                    11,13,18
    #11,13,18
    
    #stride 2/valid
    #11,13,18
    #5,6,8
    #11,13,17       !x2 + 1
    
    #stride 2/same = 1pad --> 13,15,20
    #11,13,18
    #6,7,9
    #12,14,18       !x2
    
    #same,valid: 11,6,13
    #same,same:  11,6,12
    #same, 01valid: 11,6,7,15
    #same, 01same: 11,6,7,14
    
    autoencoder = Model(inputs, x)
    return autoencoder

def conv_block(inp, filters, kernel_size, padding, trainable, channel_axis):
    x = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_initializer='he_normal', use_bias=False, trainable=trainable)(inp)
    x = BatchNormalization(axis=channel_axis, trainable=trainable)(x)
    out = Activation('relu', trainable=trainable)(x)
    return out


data = "Daten/xzt/JTE_KM3Sim_gseagen_elec-CC_3-100GeV-1_1E6-1bin-3_0gspec_ORCA115_9m_2016_100_xzt.h5"
zero_center = "Daten/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"

model_eps = "Daten/xzt/trained_vgg_3_eps_autoencoder_epoch10_supervised_up_down_epoch10.h5"
model = "Daten/xzt/trained_vgg_3_autoencoder_epoch10_supervised_up_down_epoch10.h5"

model_sup = "Daten/xzt/trained_vgg_3_supervised_up_down_epoch3.h5"
autoencoder_model = "Daten/xzt/trained_vgg_3_eps_autoencoder_epoch10.h5"
which_events = [0]


file=h5py.File(data , 'r')
zero_center_image = np.load(zero_center)
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
labels = file["y"][which_events] 
hists = file["x"][which_events]
#Get some hists from the file
hists=hists.reshape((hists.shape+(1,))).astype(np.float32)
#0 center them
centered_hists = np.subtract(hists, zero_center_image)


encoder = load_model(model)
encoder_eps = load_model(model_eps)
encoder_sup = load_model(model_sup)
autoencoder = load_model(autoencoder_model)

#Predict on 0 centered data
pred=encoder.predict_on_batch(centered_hists)
pred_eps=encoder_eps.predict_on_batch(centered_hists)


def get_out_from_layer(layer_no, model):
    get_layer_1_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_no].output])
    layer_1_output = get_layer_1_output([centered_hists,0])[0]
    return layer_1_output


def make_histogramms_of_layer(layer_no, model_1, model_2=None, title_1="Epsilon = 0.1", title_2="Epsilon = E-8"):
    #histogram of activations
    enc_feat=get_out_from_layer(layer_no, model_1)
    enc_eps_feat=get_out_from_layer(layer_no, model_2) if model_2 is not None else None
    
    plt.figure()
    
    if model_2 is not None:
        plt.subplot(121)
        plt.title(title_1)
        plt.hist(enc_feat.flatten(), 100)
        
        plt.subplot(122)
        plt.title(title_2)
        plt.hist(enc_eps_feat.flatten(), 100)
        
        plt.suptitle(model_1.layers[layer_no].name)
    
    else:
        plt.title(model_1.layers[layer_no].name)
        plt.hist(enc_feat.flatten(), 100)
        
    plt.tight_layout()

def make_weights_histogramms_of_layer(layer_no, model_1, model_2):
    #histogram of weights #layer -8
    weights = []
    for w in model_1.layers[layer_no].get_weights():
        weights.extend(w.flatten())
    weights_eps = []
    for w in model_2.layers[layer_no].get_weights():
        weights_eps.extend(w.flatten())

    plt.subplot(121)
    plt.title("Epsilon = 0.1")
    plt.hist(weights, 100)
    plt.subplot(122)
    plt.title("Epsilon = E-8")
    plt.hist(weights_eps, 100)
    plt.suptitle(model_1.layers[layer_no].name)


def make_complete_prop(model_1, save_path, model_2=None):
    with PdfPages(save_path) as pp:
        for i in range(0,len(model_1.layers)):
            make_histogramms_of_layer(i, model_1, model_2)
            pp.savefig()
            plt.close()


#make_complete_prop(model_1 = encoder, model_2=encoder_eps, save_path="vgg_3_eps_autoencoder_epoch10_supervised_up_down_epoch10_activation_1_event.pdf")
#make_histogramms_of_layer(-7, encoder, encoder_sup, "Frozen Encoder Epsilon = 0.1", "Unfrozen encoder")
make_complete_prop(autoencoder, "trained_vgg_3_eps_autoencoder_epoch10_layer_outputs.pdf")

#K.get_value(model.optimizer.lr)


#for i in range(-5,-6,-1):
#    make_histogramms_of_layer(i)
#make_weights_histogramms_of_layer()


#file=h5py.File('Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5', 'r')
#xyz_hists = np.array(file["x"]).reshape((3498,11,13,18,1))
# event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
#xyz_labels = np.array(file["y"])    



#setup_simple_model()
#setup_conv_model()
#setup_conv_model_API()

#encoder=encoded_conv_model_API()
#encoder.load_weights('../models/trained_autoencoder_test_epoch1.h5', by_name=True)
#encoder.compile(optimizer='adam', loss='mse')



"""
with open('Logfile.txt', 'w') as text_file:
    Testlog = NBatchLogger_Recent(display=1, logfile=text_file)
    Testlog = NBatchLogger_Epoch(display=1, logfile=text_file)
    history = autoencoder.fit(xyz_hists[0:100], xyz_hists[0:100], verbose=1, callbacks=[Testlog], epochs=2, batch_size=10)
"""

def plot_history():
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#plot_history()

def compare_events(no, model):
    original = xyz_hists[no].reshape(11,13,18)
    prediction = model.predict_on_batch(xyz_hists[no:(no+1)]).reshape(11,13,18)
    loss = model.evaluate(x=xyz_hists[no:(no+1)], y=xyz_hists[no:(no+1)])
    print("Loss: ",loss)
    compare_hists(original, prediction)


def plot_some_comparisons(model):
    for i in range(5):
        compare_events(i,model)
    
#autoencoder=load_model("../models/trained_autoencoder_vgg_0_epoch3.h5")

#plot_some_comparisons(autoencoder)
#compare_events(0,autoencoder)


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
        
#model = setup_vgg_like(make_autoencoder=False, modelpath_and_name="../models/trained_vgg_0_autoencoder_epoch3.h5")
#model.compile(optimizer='adam', loss='mse')

"""
train_on_batch(self, x, y, class_weight=None, sample_weight=None)

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, X_test))
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

test_on_batch(self, x, y, sample_weight=None)
predict_on_batch(self, x)
"""

