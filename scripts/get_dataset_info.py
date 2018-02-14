# -*- coding: utf-8 -*-
"""
Return a dict that contains info like path about the dataset that belongs
to a certain dataset tag.
"""

def get_dataset_info(dataset_tag):
    #Path to my Km3_net-Autoencoder folder on HPC:
    home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"
    
    #Basic default options
    #Can generate purposefully broken simulations
    #0: Normal mode
    broken_simulations_mode=0
    #How much of the file will be used for training/testing
    filesize_factor=1.0
    filesize_factor_test=1.0
    
    batchsize=32
    
    
    #Additional options are appended to the dataset tag via -XXX-YYY...
    #see list at the end of file
    splitted_dataset_tag = dataset_tag.split("-")
    dataset_tag = splitted_dataset_tag[0]
    options = splitted_dataset_tag[1:]
    
    #Sometimes, the y_values (like event energy) are not in the files (especially for autoencoders)
    #in this case, the part in the generator where those get read out can be skipped
    generator_can_read_y_values=True
    
    #Dataset to use
    if dataset_tag=="xyz":
        #Path to training and testing datafiles on HPC for xyz
        data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xyz/concatenated/"
        train_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5"
        test_data = "test_muon-CC_and_elec-CC_each_120_xyz_shuffled.h5"
        #zero_center_data = "train_muon-CC_and_elec-CC_each_480_xyz_shuffled.h5_zero_center_mean.npy"
        n_bins = (11,13,18,1)
    elif dataset_tag=="xzt":
        #for xzt
        #data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
        data_path = home_path + "data/xzt/"
        train_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
        test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
        #zero_center_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5_zero_center_mean.npy"
                
        n_bins = (11,18,50,1)
    elif dataset_tag=="xzt_new":
        #for xzt with new spatial and time binning
        data_path = home_path+"data/xzt_new_binning_spatial_time/"
        train_data = "elec-CC_and_muon-CC_xzt_train_1_to_240_shuffled_0.h5"
        test_data = "elec-CC_and_muon-CC_xzt_test_481_to_540_shuffled_0.h5"
        #zero_center_data = "" # generated automatically
        n_bins = (11,18,50,1)
        
    elif dataset_tag=="xzt_spat_tight":
        #for xzt with new spatial binning and tight time binning
        #was generated from xyzt data
        data_path = home_path+"data/xzt_new_binning_spatial_tight_time/"
        train_data = "elec-CC_and_muon-CC_xzt_train_1_to_240_shuffled_0.h5"
        test_data = "elec-CC_and_muon-CC_xzt_test_481_to_540_shuffled_0.h5"
        #zero_center_data = "" # generated automatically
        n_bins = (11,18,50,1)
        
    elif dataset_tag=="xzt_new_spatial_only":
        #for xzt with new spatial and time binning
        data_path = home_path+"data/xzt_new_binning_spatial/"
        train_data = "elec-CC_and_muon-CC_xzt_train_1_to_480_shuffled_0.h5"
        test_data = "elec-CC_and_muon-CC_xzt_test_481_to_600_shuffled_0.h5"
        #zero_center_data = "" # generated automatically
        n_bins = (11,18,50,1)
        
    elif dataset_tag=="xzt_broken":
        #for xzt
        #generates broken simulated data, very dangerous!
        #data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
        data_path = home_path+"data/xzt/"
        train_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
        test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
        n_bins = (11,18,50,1)
        
        broken_simulations_mode=1
        print("Warning: GENERATING BROKEN SIMULATED DATA")
        
    elif dataset_tag=="xzt_broken2":
        #for xzt
        #generates broken simulated data, very dangerous!
        #data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo3d/h5/xzt/concatenated/"
        data_path = home_path+"data/xzt/"
        train_data = "train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
        test_data = "test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
        n_bins = (11,18,50,1)
        
        broken_simulations_mode=2
        print("Warning: GENERATING BROKEN SIMULATED DATA")
    
    elif dataset_tag=="xyzc":
        #xyz-channel id as filter
        #11x13x18x31
        data_path = "/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/xyz_channel_-350+850/concatenated/"
        train_data = "elec-CC_and_muon-CC_xyzc_train_1_to_480_shuffled_0.h5"
        test_data = "elec-CC_and_muon-CC_xyzc_test_481_to_600_shuffled_0.h5"
        n_bins = (11,13,18,31)
        filesize_factor=0.5
        filesize_factor_test=0.5
    elif dataset_tag=="xyzc_flat":
        #original data: xyz-channel id as filter
        #11x13x18x31
        #This dataset flattens it to dimension 31 (batchsize*11*13*18, 31)
        #This means that the file actually contains 11*13*18 times more batches
        #y_values are not present
        data_path = home_path+"data/channel/"
        train_data = "elec-CC_and_muon-CC_c_train_1_to_240_shuffled_0.h5" #this is actually only 1_to_48 (fs 0.1)
        test_data = "elec-CC_and_muon-CC_c_test_481_to_540_shuffled_0.h5" # only 1_to_12
        n_bins = (31,1)
        generator_can_read_y_values=False

        
    elif dataset_tag=="debug":
        #For debug testing on my laptop:
        home_path="../"
        data_path=home_path+"Daten/"
        train_file="JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5"
        test_file="JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_588_xyz.h5"
        n_bins = (11,13,18,1)
        #file=h5py.File(train_file, 'r')
        #xyz_hists = np.array(file["x"]).reshape((3498,11,13,18,1))
        
    else:
        print("Dataset tag", dataset_tag, "is undefined!")
        raise("Dataset tag undefinded")
        
    train_file=data_path+train_data
    test_file=data_path+test_data
    #zero_center_file=data_path+zero_center_data
    
    
    #Custom options
    for option in options:
    #e.g. -filesize=0.3
        if "filesize=" in option:
            filesize_factor=float(option.split("=")[1])
            print("Filesize set to", filesize_factor)

        else:
            print("Ignoring unrecognized dataset_tag option", option)
    
    
    return_dict={}
    return_dict["home_path"]=home_path
    return_dict["train_file"]=train_file
    return_dict["test_file"]=test_file
    return_dict["n_bins"]=n_bins
    return_dict["broken_simulations_mode"]=broken_simulations_mode
    return_dict["filesize_factor"]=filesize_factor
    return_dict["filesize_factor_test"]=filesize_factor_test
    return_dict["batchsize"]=batchsize
    return_dict["generator_can_read_y_values"]=generator_can_read_y_values
    
    return return_dict
    
