
import numpy as np
import h5py
import warnings
import re
import os

from functools import reduce

train_file = "channel/elec-CC_and_muon-CC_c_train_1_to_240_shuffled_0.h5"
batchsize = 32
n_bins=(31,1)
n_gpu=1

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

train_files=((train_file, h5_get_number_of_rows(train_file)),)


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

#Copied, removed print
def get_dimensions_encoding(n_bins, batchsize):
    """
    Returns a dimensions tuple for 2,3 and 4 dimensional data.
    :param int batchsize: Batchsize that is used in generate_batches_from_hdf5_file().
    :param tuple n_bins: Declares the number of bins for each dimension (x,y,z).
                        If a dimension is equal to 1, it means that the dimension should be left out.
    :return: tuple dimensions: 2D, 3D or 4D dimensions tuple (integers).
    """
    if len(n_bins) == 2:
        #for channel data: n_bins=(31,1)
        dimensions=(batchsize, n_bins[0], n_bins[1])
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

load_zero_center_data(train_files, batchsize, n_bins, n_gpu)
