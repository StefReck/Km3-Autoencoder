# -*- coding: utf-8 -*-
import h5py
import numpy as np
from Loggers import *

"""
train_and_test_model(model, modelname, train_files, test_files, batchsize=32, n_bins=(11,13,18,1), class_type=None, xs_mean=None, epoch=0,
                         shuffle=False, lr=None, lr_decay=None, tb_logger=False, swap_4d_channels=None):
"""

def train_and_test_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
                         shuffle, lr, lr_decay, tb_logger, swap_4d_channels, save_path):
    """
    Convenience function that trains (fit_generator) and tests (evaluate_generator) a Keras model.
    For documentation of the parameters, confer to the fit_model and evaluate_model functions.
    """
    epoch += 1
    fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch, shuffle, swap_4d_channels, n_events=None, tb_logger=tb_logger, save_path=save_path)
    #fit_model speichert model ab unter ("models/trained_" + modelname + '_epoch' + str(epoch) + '.h5')
    #evaluate model evaluated und printet es in der konsole
    evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, n_events=None)

    return epoch



def fit_model(model, modelname, train_files, test_files, batchsize, n_bins, class_type, xs_mean, epoch,
              shuffle, swap_4d_channels, save_path, n_events=None, tb_logger=False):
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

    for i, (f, f_size) in enumerate(train_files):  # process all h5 files, full epoch
        if epoch > 1 and shuffle is True: # just for convenience, we don't want to wait before the first epoch each time
            print ('Shuffling file ', f, ' before training in epoch ', epoch)
            shuffle_h5(f, chunking=(True, batchsize), delete_flag=True)
        print ('Training in epoch', epoch, 'on file ', i, ',', f)

        if n_events is not None: f_size = n_events  # for testing
        
        
        with open(save_path+"models/trained_" + modelname + '_epoch' + str(epoch) + '_log.txt', 'w') as log_file:
            BatchLogger = NBatchLogger_Epoch(display=100, logfile=log_file)
            model.fit_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, f_size=f_size, zero_center_image=xs_mean, swap_col=swap_4d_channels),
                steps_per_epoch=int(f_size / batchsize), epochs=1, verbose=0, max_queue_size=10,
                validation_data=validation_data, validation_steps=validation_steps, callbacks=[BatchLogger])
            model.save(save_path+"models/trained_" + modelname + '_epoch' + str(epoch) + '.h5') #TODO
        
        
        
def evaluate_model(model, test_files, batchsize, n_bins, class_type, xs_mean, swap_4d_channels, n_events=None):
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
        print ('Testing on file ', i, ',', f)

        if n_events is not None: f_size = n_events  # for testing

        evaluation = model.evaluate_generator(
            generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, swap_col=swap_4d_channels, f_size=f_size, zero_center_image=xs_mean),
            steps=int(f_size / batchsize), max_queue_size=10)
    print ('Test sample results: ' + str(evaluation) + ' (' + str(model.metrics_names) + ')')
        
        
        
#Copied from utilities/cnn_utilities:
    
def generate_batches_from_hdf5_file(filepath, batchsize, n_bins, class_type, f_size=None, zero_center_image=None, yield_mc_info=False, swap_col=None):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated. Ideally same as the chunksize in the h5 file.
    :param tuple n_bins: Number of bins for each dimension (x,y,z,t) in the h5 file.
    :param (int, str) class_type: Tuple with the umber of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param int f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param ndarray zero_center_image: mean_image of the x dataset used for zero-centering.
    :param bool yield_mc_info: Specifies if mc-infos (y_values) should be yielded as well.
                               The mc-infos are used for evaluation after training and testing is finished.
    :param bool swap_col: Specifies, if the index of the columns for xs should be swapped. Necessary for 3.5D nets.
                          Currently available: 'yzt-x' -> [3,1,2,0] from [0,1,2,3]
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info if yield_mc_info=True).
    """
    n_bins_x, n_bins_y, n_bins_z, n_bins_t = n_bins[0], n_bins[1], n_bins[2], n_bins[3]
    #For xyz 3D Data:
    dimensions = (batchsize, n_bins_x, n_bins_y, n_bins_z, 1)

    while 1:
        f = h5py.File(filepath, "r")
        if f_size is None:
            f_size = len(f['y'])
            warnings.warn('f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                          'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')

        n_entries = 0
        while n_entries <= (f_size - batchsize):
            # create numpy arrays of input data (features)
            xs = f['x'][n_entries : n_entries + batchsize]
            xs = np.reshape(xs, dimensions).astype(np.float32)

            if swap_col is not None:
                swap_4d_channels_dict = {'yzt-x': [3,1,2,0]}
                xs[:, swap_4d_channels_dict[swap_col]] = xs[:, [0,1,2,3]]

            if zero_center_image is not None: xs = np.subtract(xs, zero_center_image) # if swap_col is not None, zero_center_image is already swapped
            # and mc info (labels)
            y_values = f['y'][n_entries:n_entries+batchsize]
            y_values = np.reshape(y_values, (batchsize, y_values.shape[1])) #TODO simplify with (y_values, y_values.shape) ?
            
            #Erstmal rausgenommen für autoencoder, denn bei mir sind die labels auch xs
            ys = 0 #np.zeros((batchsize, class_type[0]), dtype=np.float32)
            # encode the labels such that they are all within the same range (and filter the ones we don't want for now)
            """
            for c, y_val in enumerate(y_values): # Could be vectorized with numba, or use dataflow from tensorpack
                ys[c] = encode_targets(y_val, class_type)
            """
            # we have read one more batch from this file
            n_entries += batchsize

            output = (xs, xs) if yield_mc_info is False else (xs, xs) + (y_values,)
            yield output
        f.close() # this line of code is actually not reached if steps=f_size/batchsize
        
        
        
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
        
        
        
        
        
        