# -*- coding: utf-8 -*-
import h5py
import numpy as np
from keras import backend as K
import warnings
from datetime import datetime
import re
import os
from functools import reduce

from util.Loggers import NBatchLogger_Recent, NBatchLogger_Recent_Acc, NBatchLogger_Recent_CCI

#import sys
#sys.path.append('../')
#from get_dataset_info import get_dataset_info

"""
train_and_test_model(model, modelname, train_files, test_files, batchsize=32, n_bins=(11,13,18,1), class_type=None, xs_mean=None, epoch=0,
                         shuffle=False, lr=None, lr_decay=None, tb_logger=False, swap_4d_channels=None):
"""

def train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                         shuffle, lr, lr_decay, tb_logger, swap_4d_channels, save_path, is_autoencoder, verbose, broken_simulations_mode, dataset_info_dict, is_AE_adevers_training=False):
    """
    Convenience function that trains (fit_generator) and tests (evaluate_generator) a Keras model.
    For documentation of the parameters, confer to the fit_model and evaluate_model functions.
    """
    epoch += 1
    if epoch > 1 and lr_decay != 0:
        lr *= 1 - float(lr_decay)
        K.set_value(model.optimizer.lr, lr)
        print ('Set LR to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch) + ' (decay: ' + str(lr_decay) + ')')

    if broken_simulations_mode != 0:
        print("Warning: GENERATING BROKEN SIMULATED DATA\nBroken simulations mode", broken_simulations_mode )

    start_time = datetime.now()
    
    #For AAE gan training:
    if is_AE_adevers_training==1 or is_AE_adevers_training==2:
        #Train critic and generator alternating
        if epoch%2 == 0:
            #even: train only generator:
            model = freeze_adversarial_part(model, unfrozen_critic=False, unfrozen_generator=True)
            n_events = None
            is_AE_adevers_training = 2 #generator
            AE_y_labels="cat_cross"
            
        else:
            #uneven: train only critic on smaller filesize
            model = freeze_adversarial_part(model, unfrozen_critic=True, unfrozen_generator=False)
            n_events = int(train_files[0][1]/10)
            is_AE_adevers_training = 1 #critic
            AE_y_labels="cat_cross_flip"
            
    elif is_AE_adevers_training==3:
        #train only critic. Generator needs to be frozen only once
        if epoch==1:
            model = freeze_adversarial_part(model, unfrozen_critic=True, unfrozen_generator=False)
        is_AE_adevers_training = 3 #critic
        n_events = None
        AE_y_labels="wasserstein"
        
    else:
        AE_y_labels="xs"
        n_events = None
    
    
    training_hist = fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch, shuffle, swap_4d_channels, is_autoencoder=is_autoencoder, n_events=n_events, tb_logger=tb_logger, save_path=save_path, verbose=verbose, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict, AE_y_labels=AE_y_labels)
    #fit_model speichert model ab unter ("models/tag/trained_" + modelname + '_epoch' + str(epoch) + '.h5')
    #evaluate model evaluated und printet es in der konsole und in file
    end_time = datetime.now()
    time_delta=end_time-start_time
    #Elapsed time for one epoch HH::MM:SS
    elapsed_time=str(time_delta).split(".")[0]
    
    evaluation = evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, n_events=n_events, is_autoencoder=is_autoencoder, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict, AE_y_labels=AE_y_labels)

    with open(save_path+"trained_" + modelname + '_test.txt', 'a') as test_file:
        if "acc" in training_hist.history:
            #loss and accuracy
            test_file.write('\n{0}\t{1:.9g}\t{2:.9g}\t{3:.9g}\t{4:.9g}\t{5}\t{6:.4g}'.format(epoch , evaluation[0], training_hist.history["loss"][0], evaluation[1], training_hist.history["acc"][0], elapsed_time, K.get_value(model.optimizer.lr) ))
        
        elif "cat_cross_inv" in training_hist.history:
            #For the Advers AE
            test_file.write('\n{0}\t{1:.9g}\t{2:.9g}\t{3:.9g}\t{4:.9g}\t{5}\t{6:.4g}'.format(epoch , evaluation[0], training_hist.history["loss"][0], evaluation[1], training_hist.history["cat_cross_inv"][0], elapsed_time, K.get_value(model.optimizer.lr) ))
        
        else:
            #For autoencoders: only loss
            #history object looks like this: training_hist.history = {'loss': [0.9533379077911377, 0.9494166374206543]} for 2 epochs, this trains only one
            test_file.write('\n{0}\t{1:.9g}\t{2:.9g}\t{3}\t{4:.4g}'.format(epoch, evaluation, training_hist.history["loss"][0], elapsed_time, K.get_value(model.optimizer.lr) ))
    return lr

def fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
              shuffle, swap_4d_channels, save_path, is_autoencoder, verbose, broken_simulations_mode, dataset_info_dict, n_events=None, tb_logger=False,AE_y_labels="xs"):
    """
    Trains a model based on the Keras fit_generator method.
    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the test_files is used.
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param str modelname: Name of the model.
    :param list train_files: list of tuples that contains the testfiles and their number of rows (filepath, f_size).
    :param list test_files: list of tuples that contains the testfiles and their number of rows for the tb_callback.
    :param int batchsize: Batchsize that is used in the fit_generator method.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in both the train- and test_files.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param int epoch: Epoch of the model if it has been trained before.
    :param bool shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for training.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param bool tb_logger: Declares if a tb_callback during fit_generator should be used (takes long time to save the tb_log!).
    """

    validation_data, validation_steps, callbacks = None, None, None

    history = None
    
    metrics = model.metrics_names #e.g. ['loss', 'acc'] or ['loss']
    
    for i, (f, f_size) in enumerate(train_files):  # process all h5 files, full epoch
        if epoch > 1 and shuffle is True: # just for convenience, we don't want to wait before the first epoch each time
            print ('Shuffling file ', f, ' before training in epoch ', epoch)
            shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)
        print ('Training in epoch', epoch, 'on train file ', i, "at", datetime.now().strftime('%H:%M:%S'))

        if n_events is not None: f_size = n_events  # for testing
        
        
        with open(save_path+"trained_" + modelname + '_epoch' + str(epoch) + '_log.txt', 'w') as log_file:
            
            if "acc" in metrics:
                BatchLogger = NBatchLogger_Recent_Acc(display=500, logfile=log_file)
            elif "cat_cross_inv" in metrics:
                BatchLogger = NBatchLogger_Recent_CCI(display=500, logfile=log_file)
            else:
                BatchLogger = NBatchLogger_Recent(display=500, logfile=log_file)
                
            history = model.fit_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, is_autoencoder=is_autoencoder, f_size=f_size, zero_center_image=xs_mean, swap_col=swap_4d_channels, broken_simulations_mode=broken_simulations_mode, is_in_test_mode=False, dataset_info_dict=dataset_info_dict, AE_y_labels=AE_y_labels),
                steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=verbose, max_queue_size=10,
                validation_data=validation_data, validation_steps=validation_steps, callbacks=[BatchLogger],)
            model.save(save_path+"trained_" + modelname + '_epoch' + str(epoch) + '.h5') #TODO
        
    return history
        
        
def evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, is_autoencoder, broken_simulations_mode, dataset_info_dict, n_events=None,AE_y_labels="xs"):
    """
    Evaluates a model with validation data based on the Keras evaluate_generator method.
    :param ks.model.Model/Sequential model: Keras model (trained) of a neural network.
    :param list test_files: list of tuples that contains the testfiles and their number of rows.
    :param int batchsize: Batchsize that is used in the evaluate_generator method.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the test_files.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for evaluating.
    """
    for i, (f, f_size) in enumerate(test_files):
        print ('Testing on test file ', i, "at", datetime.now().strftime('%H:%M:%S'))

        if n_events is not None: f_size = n_events  # for testing

        evaluation = model.evaluate_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, is_autoencoder=is_autoencoder, swap_col=swap_4d_channels, f_size=f_size, zero_center_image=xs_mean, broken_simulations_mode=broken_simulations_mode, is_in_test_mode=True, dataset_info_dict=dataset_info_dict, AE_y_labels=AE_y_labels),
            steps=int(f_size / batchsize), max_queue_size=10)
    print('Test sample results: ', str(evaluation), ' (', str(model.metrics_names), ')')
    return evaluation

def freeze_adversarial_part(model, unfrozen_generator, unfrozen_critic):
    #Go through the network and freeze/unfreeze layers according to vars
    last_layer_of_AE_name = "AE_output_layer"
    
    ae_loss=model.loss
    pre_optimizer = model.optimizer
    pre_metrics= model.metrics
    
    is_in_generator_part=True
    for layer in model.layers:
        if is_in_generator_part:
            layer.trainable=unfrozen_generator
        else:
            layer.trainable=unfrozen_critic
        
        if layer.name == last_layer_of_AE_name:
            is_in_generator_part=False
            
    model.compile(loss=ae_loss, optimizer=pre_optimizer, metrics=pre_metrics)
    
    print("State of network: Generator unfrozen:", unfrozen_generator, "Critic unfrozen:", unfrozen_critic)
    
    return model

def modify_batches(xs, batchsize, dataset_info_dict, zero_center_image, y_values=None):
    """
    Makes changes to the data as read from h5 file, e.g. adding noise, ...
        xs:         Batch of event data, shape (batchsize, xzt, 1)
        y_values:   The mc_info vector, containing all infos about the event.
    """
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] 
    
    if broken_simulations_mode==1:
        #encode up-down info in the first bin
        #y_values are needed for this
        
        # if swap_col is not None, zero_center_image is already swapped
        if zero_center_image is not None: xs = np.subtract(xs, zero_center_image)
        
        ys = np.zeros((batchsize, 1), dtype=np.float32)
        # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
        for c, y_val in enumerate(y_values): # Could be vectorized with numba, or use dataflow from tensorpack
            ys[c] = encode_targets(y_val, class_type=(1, "up_down"))
            #ys is same length as xs and contains 1 (for up) or 0 (for down), shape (batchsize,1)
        ys=ys.flatten()*2 - 1 #now ys is -1 for down and 1 for up, shape (batchsize,)
            
        for i in range(len(xs)):
            xs[i].itemset(0,ys[i])
            
    elif broken_simulations_mode==2:
        #randomly add poisson distributed noise
        #rauschen bisherige rate: 10kHz pro pmt
        #Zusätzliches rauschen z.B. 5 kHz
        #Zeitfenster 1100ns etwa
        #31 pmts pro dom
        #Erwartungswert rauschen für ganzes Zeitfenster pro DOM:
        #5kHz * 1100ns * 31 pmts = 0.1705
        #Erwartungswert pro Zeitbin und DOM: 0.1705 / 50 = 0.00341
        #Aufsummiert über y (13 bins): 0.00341*13=0.04433
        
        # if swap_col is not None, zero_center_image is already swapped
        if zero_center_image is not None: xs = np.subtract(xs, zero_center_image)
        
        poisson_noise_expectation_value=0.04433 #5kHz
        #zero centered noise:
        noise = np.random.poisson(2*poisson_noise_expectation_value, size=xs.shape) - 2*poisson_noise_expectation_value
        #only add noise to the bins that are not always empty. 
        #xs = xs + np.multiply(noise[np.tile(zero_center_image!=0, (batchsize,1,1))], noise)
        #For xzt though, this is unecessary since there are no empty bins
        xs = xs + noise
        
    elif broken_simulations_mode==3:
        #replace the lower third (z) of the measured signal with 0
        #xs.shape: (32,11,18,50,1)
        
        # if swap_col is not None, zero_center_image is already swapped
        if zero_center_image is not None: xs = np.subtract(xs, zero_center_image)
        
        xs[:,:,:6,:,:]=np.zeros_like(xs[:,:,:6,:,:])
        
    elif broken_simulations_mode==4:
        #increase count rate of down-going events
        #y_values are needed for this; xs_mean is not updated
        ys = np.zeros((batchsize, 1), dtype=np.float32)
        # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
        for c, y_val in enumerate(y_values): # Could be vectorized with numba, or use dataflow from tensorpack
            ys[c] = encode_targets(y_val, class_type=(1, "up_down"))
            #ys is same length as xs and contains 1 (for up) or 0 (for down), shape (batchsize,1)
        ys=ys.flatten().astype(bool) #now ys is shape (batchsize,); 1 (for up) or 0 (for down)
        
        #where are more then x hits
        hits_mask = xs>=3
        #which events are up going (this is equal to ys)
        #up_going = ys==1
        #where are more then x hits AND the event is down going
        hits_mask[ys] = np.zeros_like(hits_mask[ys])
        #expectation value = 2, with n=3,p=2/3
        multiply_by = np.random.binomial(3,2/3, hits_mask.shape)
        xs = np.add(xs, hits_mask*multiply_by)
            
        # if swap_col is not None, zero_center_image is already swapped
        if zero_center_image is not None: xs = np.subtract(xs, zero_center_image)
        
        #Broken 5 mode was precalculated in its own files
    
    elif broken_simulations_mode==12:
        raise NameError("Broken mode 12 is not generated on the fly anymore, use the generated files instead!")
        #Add poisson noise that is proportional to 1/energy
        #y_values are needed for this
        #Zero center first; the noise that will be added is 0 centered itself
        if zero_center_image is not None: xs = np.subtract(xs, zero_center_image)
        #Get the mc energy of the events:
        ys = y_values[:,2]
        
        #poisson_noise_expectation_value=0.08866 for 10kHz noise
        
        """This expectation value is proportional to 1/mc_energy, 
        ranging between 10 kHz at 3 GeV and 0.3 kHz at 100 GeV
        this means that there is little change happening after 20 GeV or so"""
        #poisson_noise_expectation_value = 0.08866 * 3/ys
        
        """Expectation value linearly decreasing from 10 kHz at 3 GeV to 0 kHz at 100 GeV"""
        poisson_noise_expectation_value = 0.08866 * (100-ys)/97
        
        #zero centered noise, has the shape (dims, batchsize), while xs has the shape (batchsize, dims)
        noise = np.random.poisson(poisson_noise_expectation_value, size=xs.shape[1:]+ys.shape) - poisson_noise_expectation_value
        #permute so that noise has shape (batchsize, dims), just as xs
        noise = np.transpose(noise, np.roll(np.arange(len(noise.shape)), shift=1) )
        
        xs = xs + noise
        
    else:
        # if swap_col is not None, zero_center_image is already swapped
        if zero_center_image is not None: xs = np.subtract(xs, zero_center_image)
        
    return xs

#Copied from cnn_utilities and modified:
def generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, is_autoencoder, dataset_info_dict, broken_simulations_mode=0, f_size=None, zero_center_image=None, yield_mc_info=False, swap_col=None, is_in_test_mode = False, AE_y_labels="xs"):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated. Ideally same as the chunksize in the h5 file.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the h5 file.
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    broken_simulations_mode: Generate Simulations that are purpusefully suboptimal
                                0: Normal simulation (Not broken)
                                1: The up-down info is encoded in the first bin
                                2: Add noise to data. Seed is set manually, so that the images always have the same noise.
    :param int f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param ndarray zero_center_image: mean_image of the x dataset used for zero-centering.
    :param bool yield_mc_info: Specifies if mc-infos (y_values) should be yielded as well.
                               The mc-infos are used for evaluation after training and testing is finished.
    :param bool swap_col: Specifies, if the index of the columns for xs should be swapped. Necessary for 3.5D nets.
                          Currently available: 'yzt-x' -> [3,1,2,0] from [0,1,2,3]
    :param bool is_in_test_mode: Is this used in testing a model? Only used for random seed initialization for broken data mode 2.
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info if yield_mc_info=True).
    """
    #flatten_to_filter=dataset_info_dict["flatten_to_filter"]
    
    dimensions = get_dimensions_encoding(n_bins, batchsize)
    
    if broken_simulations_mode == 2 or broken_simulations_mode == 4 or broken_simulations_mode == 12:
        #make it so the noise is always the same on the same histogramm
        #also use different noise on test and train data
        if is_in_test_mode == False:
            np.random.seed(100)
        else:
            np.random.seed(101)


    while 1:
        f = h5py.File(filepath, "r")
        if f_size is None:
            f_size = len(f['x'])
            warnings.warn('f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                          'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')
        n_entries = 0
        while n_entries <= (f_size - batchsize):
            # create numpy arrays of input data (features)
            xs = f['x'][n_entries : n_entries + batchsize] #(batchsize, n_bins)
            xs = np.reshape(xs, dimensions).astype(np.float32)
            
            #if swap_col is not None:
            #    swap_4d_channels_dict = {'yzt-x': [3,1,2,0]}
            #    xs[:, swap_4d_channels_dict[swap_col]] = xs[:, [0,1,2,3]]

            if dataset_info_dict["generator_can_read_y_values"]==True:
                # and mc info (labels)
                y_values = f['y'][n_entries:n_entries+batchsize]
                y_values = np.reshape(y_values, (batchsize, y_values.shape[1])) #TODO simplify with (y_values, y_values.shape) ?
            else:
                y_values = None
                
            #Make changes, e.g. add broken mode, subtract zero_center_image, ...
            xs = modify_batches(xs=xs, y_values=y_values, batchsize=batchsize, dataset_info_dict=dataset_info_dict, zero_center_image=zero_center_image)
            
            #Modified for autoencoder:
            if is_autoencoder == True:
                if AE_y_labels == "cat_cross":
                    #For critic training
                    #1,0 means fake, 0,1 means real
                    labels = np.repeat([[[1,0],[0,1]],],xs.shape[0],0)
                elif AE_y_labels == "cat_cross_flip":
                    #for generator training
                    #flip the labels, so that the generator learns to maximize the loss!
                    #1,0 means fake, 0,1 means real
                    labels = np.repeat([[[0,1],[1,0]],],xs.shape[0],0)
                    
                elif AE_y_labels == "wasserstein":
                    #For critic training
                    #-1 means fake, 1 means real
                    labels = np.repeat([[[-1],[1]],],xs.shape[0],0)
                elif AE_y_labels == "wasserstein_flip":
                    #for generator training
                    #flip the labels, so that the generator learns to maximize the loss!
                    #1,0 means fake, 0,1 means real
                    labels = np.repeat([[[1],[-1]],],xs.shape[0],0)
                    
                else:
                    #MSE Autoencoder
                    labels = xs
                output = (xs, labels) if yield_mc_info is False else (xs, labels) + (y_values,)
                
            else:
                ys = np.zeros((batchsize, class_type[0]), dtype=np.float32)
                # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
                for c, y_val in enumerate(y_values): # Could be vectorized with numba, or use dataflow from tensorpack
                    ys[c] = encode_targets(y_val, class_type)
                output = (xs, ys) if yield_mc_info is False else (xs, ys) + (y_values,)


            # we have read one more batch from this file
            n_entries += batchsize
            yield output
            
        f.close() # this line of code is actually not reached if steps=f_size/batchsize
       
#For testing:
"""
f="../Daten/JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5"
batchsize=32
n_bins=(11,13,18,1)
class_type=(2,"up_down")
is_autoencoder=True
f_size=None
xs_mean=None
broken_simulations_mode=0
dataset_info_dict={}
dataset_info_dict["flatten_to_filter"]=True
dataset_info_dict["broken_simulations_mode"]=broken_simulations_mode
g=generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, is_autoencoder=is_autoencoder, f_size=f_size, zero_center_image=xs_mean, broken_simulations_mode=broken_simulations_mode, is_in_test_mode=False, dataset_info_dict=dataset_info_dict)
"""
        
        
#Copied, removed print
def get_dimensions_encoding(n_bins, batchsize):
    """
    Returns a dimensions tuple for 2,3 and 4 dimensional data.
    :param int batchsize: Batchsize that is used in generate_batches_from_hdf5_file().
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z).
                        If a dimension is equal to 1, it means that the dimension should be left out.
    :return: tuple dimensions: 2D, 3D or 4D dimensions tuple (integers).
    """
    if len(n_bins) == 1:
        #for channel data: n_bins=(31,)
        dimensions=(batchsize, n_bins[0],)
    else:
        #e.g. xzt n_bins=(11,18,50,1)
        n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
        if n_bins_x == 1:
            if n_bins_y == 1:
                #print 'Using 2D projected data without dimensions x and y'
                dimensions = (batchsize, n_bins_z, n_bins_t, 1)
            elif n_bins_z == 1:
                #print 'Using 2D projected data without dimensions x and z'
                dimensions = (batchsize, n_bins_y, n_bins_t, 1)
            elif n_bins_t == 1:
                #print 'Using 2D projected data without dimensions x and t'
                dimensions = (batchsize, n_bins_y, n_bins_z, 1)
            else:
                #print 'Using 3D projected data without dimension x'
                dimensions = (batchsize, n_bins_y, n_bins_z, n_bins_t, 1)
    
        elif n_bins_y == 1:
            if n_bins_z == 1:
                #print 'Using 2D projected data without dimensions y and z'
                dimensions = (batchsize, n_bins_x, n_bins_t, 1)
            elif n_bins_t == 1:
                #print 'Using 2D projected data without dimensions y and t'
                dimensions = (batchsize, n_bins_x, n_bins_z, 1)
            else:
                #print 'Using 3D projected data without dimension y'
                dimensions = (batchsize, n_bins_x, n_bins_z, n_bins_t, 1)
    
        elif n_bins_z == 1:
            if n_bins_t == 1:
                #print 'Using 2D projected data without dimensions z and t'
                dimensions = (batchsize, n_bins_x, n_bins_y, 1)
            else:
                #print 'Using 3D projected data without dimension z'
                dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_t, 1)
    
        elif n_bins_t == 1:
            #print 'Using 3D projected data without dimension t'
            dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, 1)
    
        else:
            #print 'Using full 4D data'
            dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, n_bins_t)

    return dimensions

        
#Copied unchanged from cnn_utilities
def encode_targets(y_val, class_type):
    """
    Encodes the labels (classes) of the images.
    :param ndarray(ndim=1) y_val: Array that contains ALL event class information for one event.
           ---------------------------------------------------------------------------------------------------------------------------
           Current content: [event_id -> 0, particle_type -> 1, energy -> 2, isCC -> 3, bjorkeny -> 4, dir_x/y/z -> 5/6/7, time -> 8]
           ---------------------------------------------------------------------------------------------------------------------------
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :return: ndarray(ndim=1) train_y: Array that contains the encoded class label information of the input event.
    """
    def get_class_up_down_categorical(dir_z, n_neurons):
        """
        Converts the zenith information (dir_z) to a binary up/down value.
        :param float32 dir_z: z-direction of the event_track (which contains dir_z).
        :param int n_neurons: defines the number of neurons in the last cnn layer that should be used with the categorical array.
        :return ndarray(ndim=1) y_cat_up_down: categorical y ('label') array which can be fed to a NN.
                                               E.g. [0],[1] for n=1 or [0,1], [1,0] for n=2
        """
        # analyze the track info to determine the class number
        up_down_class_value = int(np.sign(dir_z)) # returns -1 if dir_z < 0, 0 if dir_z==0, 1 if dir_z > 0

        if up_down_class_value == 0:
            #print ('Warning: Found an event with dir_z==0. Setting the up-down class randomly.')
            #TODO maybe [0.5, 0.5], but does it make sense with cat_crossentropy?
            up_down_class_value = np.random.randint(2)

        if up_down_class_value == -1: up_down_class_value = 0 # Bring -1,1 values to 0,1

        y_cat_up_down = np.zeros(n_neurons, dtype='float32')

        if n_neurons == 1:
            y_cat_up_down[0] = up_down_class_value # 1 or 0 for up/down
        else:
            y_cat_up_down[up_down_class_value] = 1 # [0,1] or [1,0] for up/down

        return y_cat_up_down


    def convert_particle_class_to_categorical(particle_type, is_cc, num_classes=4):
        """
        Converts the possible particle types (elec/muon/tau , NC/CC) to a categorical type that can be used as tensorflow input y
        :param int particle_type: Specifies the particle type, i.e. elec/muon/tau (12, 14, 16). Negative values for antiparticles.
        :param int is_cc: Specifies the interaction channel. 0 = NC, 1 = CC.
        :param int num_classes: Specifies the total number of classes that will be discriminated later on by the CNN. I.e. 2 = elec_NC, muon_CC.
        :return: ndarray(ndim=1) categorical: returns the categorical event type. I.e. (particle_type=14, is_cc=1) -> [0,0,1,0] for num_classes=4.
        """
        if num_classes == 4:
            particle_type_dict = {(12, 0): 0, (12, 1): 1, (14, 1): 2, (16, 1): 3}  # 0: elec_NC, 1: elec_CC, 2: muon_CC, 3: tau_CC
        else:
            raise ValueError('A number of classes !=4 is currently not supported!')

        category = int(particle_type_dict[(abs(particle_type), is_cc)])
        categorical = np.zeros(num_classes, dtype='int8') # TODO try bool
        categorical[category] = 1

        return categorical

    if class_type[1] == 'muon-CC_to_elec-NC':
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(class_type[0], dtype='float32') # 1 ([0], [1]) or 2 ([0,1], [1,0]) neurons

        if class_type[0] == 1: # 1 neuron
            if categorical_type[2] != 0:
                train_y[0] = categorical_type[2] # =0 if elec-NC, =1 if muon-CC

        else: # 2 neurons
            assert class_type[0] == 2
            train_y[0] = categorical_type[0]
            train_y[1] = categorical_type[2]

    elif class_type[1] == 'muon-CC_to_elec-CC':
        categorical_type = convert_particle_class_to_categorical(y_val[1], y_val[3], num_classes=4)
        train_y = np.zeros(class_type[0], dtype='float32')

        if class_type[0] == 1: # 1 neuron
            if categorical_type[2] != 0:
                train_y[0] = categorical_type[2] # =0 if elec-CC, =1 if muon-CC

        else: # 2 neurons
            assert class_type[0] == 2
            train_y[0] = categorical_type[1]
            train_y[1] = categorical_type[2]

    elif class_type[1] == 'up_down':
        #supports both 1 or 2 neurons at the cnn softmax end
        train_y = get_class_up_down_categorical(y_val[7], class_type[0])
        
    elif class_type[1] == 'energy':
        #always has one neuron as the output
        train_y = y_val[2:3] # energy

    else:
        print ("Class type " + str(class_type) + " not supported!")
        return y_val

    return train_y
        
        
#Kopiert von utilities/input utilities:
def h5_get_number_of_rows(h5_filepath):
    """
    Gets the total number of rows of the first dataset of a .h5 file. Hence, all datasets should have the same number of rows!
    :param string h5_filepath: filepath of the .h5 file.
    :return: int number_of_rows: number of rows of the .h5 file in the first dataset.
    """
    f = h5py.File(h5_filepath, 'r')
    #Bug?
    #number_of_rows = f[f.keys()[0]].shape[0]
    number_of_rows = f["x"].shape[0]
    f.close()

    return number_of_rows
        
        
#Kopiert von cnn_utilities
def load_zero_center_data(train_files, batchsize, n_bins, n_gpu):
    """
    Gets the xs_mean array that can be used for zero-centering.
    The array is either loaded from a previously saved file or it is calculated on the fly.
    Currently only works for a single input training file!
    :param list((train_filepath, train_filesize)) train_files: list of tuples that contains the trainfiles and their number of rows.
    :param int batchsize: Batchsize that is being used in the data.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the tran_file.
    :param int n_gpu: Number of gpu's, used for calculating the available RAM space in get_mean_image().
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """
    if len(train_files) > 1:
        warnings.warn('More than 1 train file for zero-centering is currently not supported! '
                      'Only the first file is used for calculating the xs_mean_array.')

    filepath = train_files[0][0]

    # if the file has a shuffle index (e.g. shuffled_6.h5) and a .npy exists for the first shuffled file (shuffled.h5), we don't want to calculate the mean again
    shuffle_index = re.search('shuffled(.*).h5', filepath)
    filepath_without_index = re.sub(shuffle_index.group(1), '', filepath)

    if os.path.isfile(filepath_without_index + '_zero_center_mean.npy') is True:
        print ('Loading existing zero center image:', filepath_without_index + '_zero_center_mean.npy')
        xs_mean = np.load(filepath_without_index + '_zero_center_mean.npy')
    else:
        print ('Calculating the xs_mean_array in order to zero_center the data!')
        dimensions = get_dimensions_encoding(n_bins, batchsize)
        xs_mean = get_mean_image(filepath, filepath_without_index, dimensions, n_gpu)

    return xs_mean


def get_mean_image(filepath, filepath_without_index, dimensions, n_gpu):
    """
    Returns the mean_image of a xs dataset.
    Calculating still works if xs is larger than the available memory and also if the file is compressed!
    :param str filepath: Filepath of the data upon which the mean_image should be calculated.
    :param str filepath_without_index: filepath without the number index.
    :param tuple dimensions: Dimensions tuple for 2D, 3D or 4D data.
    :param filepath: Filepath of the input data, used as a str for saving the xs_mean_image.
    :param int n_gpu: Number of used gpu's that is related to how much RAM is available (16G per GPU).
    :return: ndarray xs_mean: mean_image of the x dataset. Can be used for zero-centering later on.
    """
    f = h5py.File(filepath, "r")

    # check available memory and divide the mean calculation in steps
    total_memory = n_gpu * 8e9 # In bytes. Take 1/2 of what is available per GPU (16G), just to make sure.
    #filesize = os.path.getsize(filepath) # doesn't work for compressed files
    filesize =  get_array_memsize(f['x'])

    steps = int(np.ceil(filesize/total_memory))*10
    n_rows = f['x'].shape[0]
    stepsize = int(n_rows / float(steps))

    xs_mean_arr = None
    print("Shape of file:", f['x'].shape, "Steps:", steps, "Stepsize:", stepsize)
    for i in range(steps):
        print ('Calculating the mean_image of the xs dataset in step ' + str(i))
        if xs_mean_arr is None: # create xs_mean_arr that stores intermediate mean_temp results
            xs_mean_arr = np.zeros((steps, ) + f['x'].shape[1:], dtype=np.float64)

        if i == steps-1 or steps == 1: # for the last step, calculate mean till the end of the file
            xs_mean_temp = np.mean(f['x'][i * stepsize: n_rows], axis=0, dtype=np.float64)
        else:
            xs_mean_temp = np.mean(f['x'][i*stepsize : (i+1) * stepsize], axis=0, dtype=np.float64)
        xs_mean_arr[i] = xs_mean_temp

    xs_mean = np.mean(xs_mean_arr, axis=0, dtype=np.float64).astype(np.float32)
    xs_mean = np.reshape(xs_mean, dimensions[1:]) # give the shape the channels dimension again if not 4D

    np.save(filepath_without_index + '_zero_center_mean.npy', xs_mean)
    print("New zero center image saved as", filepath_without_index + '_zero_center_mean.npy')
    return xs_mean

def get_array_memsize(array):
    """
    Calculates the approximate memory size of an array.
    :param ndarray array: an array.
    :return: float memsize: size of the array in bytes.
    """
    shape = array.shape
    n_numbers = reduce(lambda x, y: x*y, shape) # number of entries in an array
    precision = 8 # Precision of each entry, typically uint8 for xs datasets
    memsize = (n_numbers * precision) / float(8) # in bytes

    return memsize
        
        






