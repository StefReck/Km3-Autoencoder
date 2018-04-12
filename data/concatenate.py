# -*- coding: utf-8 -*-
"""
Concatenate two big files.
"""

import h5py
import numpy as np
import math

path="/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/stefan_xzt-c_50b_w-out-geo-fix_timeslice-relative/concatenated/"
files = ["elec-CC_and_muon-CC_xyzt_train_1_to_120_shuffled_0.h5",
         "elec-CC_and_muon-CC_xyzt_train_121_to_240_shuffled_0.h5",]


new_file= "/home/woody/capn/mppi013h/Km3-Autoencoder/data/xztc/elec-CC_and_muon-CC_xyzt_train_1_to_240_shuffled_0.h5"




def get_cum_number_of_rows(file_list, cuts=False):
    """
    Returns the cumulative number of rows (axis_0) in a list based on the specified input .h5 files.
    This information is needed for concatenating the .h5 files later on.
    Additionally, the average number of rows for all the input files is calculated, in order to derive a sensible chunksize (optimized for diskspace).
    :param list file_list: list that contains all filepaths of the input files.
    :param bool cuts: specifies if cuts should be used for getting the cum_number_of_rows.
                      In this case, the function also returns an events_survive_cut dict that specifies, if an event in a certain file survives the cut.
    :return: list cum_number_of_rows_list: list that contains the cumulative number of rows (i.e. [0,100,200,300,...] if each file has 100 rows).
    :return: int mean_number_of_rows: specifies the average number of rows (rounded up to int) for the files in the file_list.
    :return: None/dict dict_events_survive_cut: None if cuts=False, else it contains a dict with the information which events in a certain h5 file survive the cuts.
    """
    total_number_of_rows = 0
    cum_number_of_rows_list = [0]
    number_of_rows_list = []  # used for approximating the chunksize
    dict_events_survive_cut = None

    for file_name in file_list:
        f = h5py.File(file_name, 'r')
        # get number of rows from the first folder of the file -> each folder needs to have the same number of rows
        
        total_number_of_rows += f[list(f.keys())[0]].shape[0]
        cum_number_of_rows_list.append(total_number_of_rows)
        number_of_rows_list.append(f[list(f.keys())[0]].shape[0])

        f.close()

    mean_number_of_rows = math.ceil(np.mean(number_of_rows_list))

    return cum_number_of_rows_list, mean_number_of_rows, dict_events_survive_cut



def concatenate_h5_files(files, new_file):
    """
    Main code. Concatenates .h5 files with multiple datasets, where each dataset in one file needs to have the same number of rows (axis_0).
    Gets user input with aid of the parse_input() function. By default, the chunksize for the output .h5 file is automatically computed.
    based on the average number of rows per file, in order to eliminate padding (wastes disk space).
    For faster I/O, the chunksize should be set by the user depending on the use case.
    In deep learning applications for example, the chunksize should be equal to the batch size that is used later on for reading the data.
    """
    
    file_list = files
    output_filepath=new_file
    custom_chunksize=(True, 32)
    compress=('gzip', 1)
    
    #file_list, output_filepath, custom_chunksize, compress, cuts = parse_input()
    cum_rows_list, mean_number_of_rows, dict_events_survive_cut = get_cum_number_of_rows(file_list, cuts=False)

    
    file_output = h5py.File(output_filepath, 'w')

    for n, input_file_name in enumerate(file_list):

        print ('Processing file ' + file_list[n])

        input_file = h5py.File(input_file_name, 'r')

        for folder_name in input_file:

            folder_data = input_file[folder_name]

            print ('Shape and dtype of dataset ' + folder_name + ': ' + str(folder_data.shape) + ' ; ' + str(folder_data.dtype))

            if n == 0:
                # first file; create the dummy dataset with no max shape
                maxshape = (None,) + folder_data.shape[1:] # change shape of axis zero to None
                chunks = (custom_chunksize[1],) + folder_data.shape[1:] if custom_chunksize[0] is True else (mean_number_of_rows,) + folder_data.shape[1:]

                output_dataset = file_output.create_dataset(folder_name, data=folder_data, maxshape=maxshape, chunks=chunks,
                                                            compression=compress[0], compression_opts=compress[1])

                output_dataset.resize(cum_rows_list[-1], axis=0)

            else:
                file_output[folder_name][cum_rows_list[n]:cum_rows_list[n+1]] = folder_data

        file_output.flush()

    print ('Output information:')
    print ('-------------------')
    print ('The output file contains the following datasets:')
    for folder_name in file_output:
        print ('Dataset ' + folder_name + ' with the following shape, dtype and chunks (first argument is the chunksize in axis_0): \n' /
              +  str(file_output[folder_name].shape) + ' ; ' + str(file_output[folder_name].dtype) + ' ; ' + str(file_output[folder_name].chunks))

    file_output.close()

files=[path+file for file in files]
concatenate_h5_files(files, new_file)

