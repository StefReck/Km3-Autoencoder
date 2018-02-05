# -*- coding: utf-8 -*-
"""
Return a dict that contains info like path about the dataset that belongs
to a certain dataset tag.
"""

def get_dataset_info(dataset_tag):
    #Path to my Km3_net-Autoencoder folder on HPC:
    home_path="/home/woody/capn/mppi013h/Km3-Autoencoder/"
    
    #Can generate purposefully broken simulations
    #0: Normal mode
    broken_simulations_mode=0
    #1: encode up-down info into first bin
    
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
    
    return_dict={}
    return_dict["home_path"]=home_path
    return_dict["train_file"]=train_file
    return_dict["test_file"]=test_file
    return_dict["n_bins"]=n_bins
    return_dict["broken_simulations_mode"]=broken_simulations_mode
    
    return return_dict
    