import h5py
import numpy as np

original_train_file="/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/xyz_channel_-350+850/concatenated/elec-CC_and_muon-CC_xyzc_train_1_to_480_shuffled_0.h5"
original_test_file ="/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/xyz_channel_-350+850/concatenated/elec-CC_and_muon-CC_xyzc_test_481_to_600_shuffled_0.h5"

#outfile_train="channel/elec-CC_and_muon-CC_c_train_1_to_240_shuffled_0.h5"
#outfile_test= "channel/elec-CC_and_muon-CC_c_test_481_to_540_shuffled_0.h5"

outfile_train="channel/elec-CC_and_muon-CC_c_event_train_1_to_240_shuffled_0.h5"
outfile_test ="channel/elec-CC_and_muon-CC_c_event_test_481_to_540_shuffled_0.h5"
   
# x, y, z, t, c
#11,13,18,50,31

#11,13,18,31

#percentage of events to keep
fraction=0.5
#which axis including filesize to sum over, None if no sum
#e.g. X,11,13,18,50 --> X,11,18,50 axis=2
sum_over_axis=None
#
reshape_to_channel_and_shuffle=True
only_doms_with_more_then=1

#show how often certain total number of hits in a dom occur; No files will be generated
make_channel_statistics = True

def generate_file(file, save_to, fraction, sum_over_axis, reshape_to_channel_and_shuffle, only_doms_with_more_then):     
    print("Generating file", save_to)
    f=h5py.File(file, "r")
    shape=f["x"].shape #e.g. (X,11,13,18,50)
    print("Original shape of hists:", shape)
    up_to_which=int(shape[0]*fraction)
    print("Events left after cut:", up_to_which)

    if sum_over_axis is not None:
        hists=np.sum(f["x"][:up_to_which], axis=sum_over_axis)
    else:
        hists=f["x"][:up_to_which]
        
        
    if reshape_to_channel_and_shuffle == True:
        hists = hists.reshape(-1, hists.shape[-1]) # dimension e.g. (X*11*13*18, 50)
        np.random.shuffle(hists)
        
        if only_doms_with_more_then > 0:
            #this will delete doms with less then ... hits, so that the ratio of 
            #doms with more/less hits is at a defined value
            ratio_of_more_to_less = 2 #how many with more hits come on one with less
            #total number of hits of a dom
            sum_of_all_channels = np.sum(hists, axis=1) # e.g. (X*11*13*18,)
            #how many doms have more then ... hits
            how_many_with_more_hits = np.sum(sum_of_all_channels>only_doms_with_more_then)
            #how many doms under this threshold to keep
            how_many_with_less_hits_to_keep = int(how_many_with_more_hits/ratio_of_more_to_less)
            #which doms have less hits then the threshold
            where_hists_with_less_hits = np.where(sum_of_all_channels<=only_doms_with_more_then)[0]
            #delete some of those doms that are under the threshold
            delete_these = where_hists_with_less_hits[how_many_with_less_hits_to_keep:]
            
            print("There are", how_many_with_more_hits, "doms with more then", only_doms_with_more_then, "hits, and there will be", how_many_with_less_hits_to_keep, " with less hits left after this.")
            
            hists = np.delete(hists, delete_these, axis=0)
        
        mc_infos=np.zeros(100)
    else:
        mc_infos=f["y"][:up_to_which]
        
    print("New shape (hists):", hists.shape, ", mc infos:", mc_infos.shape)
    store_histograms_as_hdf5(hists, mc_infos, save_to, compression=("gzip", 1))


def make_channel_statistics(file):
    f=h5py.File(file, "r")
    shape=f["x"].shape #e.g. (X,11,13,18,50)
    print("Original shape of hists:", shape)
    up_to_which=int(shape[0]*0.5)
    hists=f["x"][:up_to_which]
    #number of occurances of total hits in a whole dom
    count_number = np.bincount(np.sum(hists, axis=-1).flatten())
    print("How many DOMs are there with how many total hits:")
    for no_counts in range(len(count_number)):
        print(no_counts, " Hits:", count_number[no_counts], "times")
        

def store_histograms_as_hdf5(hists, mc_infos, filepath_output, compression=(None, None)):
    """
    Takes numpy histograms ('images') for a certain projection as well as the mc_info ('tracks') and saves them to a h5 file.
    :param ndarray(ndim=2) hists: array that contains all histograms for a certain projection.
    :param ndarray(ndim=2) mc_infos: 2D array containing important MC information for each event_id. [event_id, particle_type, energy, isCC, categorical event types]
    :param str filepath_output: complete filepath of the created h5 file.
    :param (None/str, None/int) compression: Tuple that specifies if a compression filter should be used. Either ('gzip', 1-9) or ('lzf', None).
    """

    h5f = h5py.File(filepath_output, 'w')

    chunks_hists = (32,) + hists.shape[1:] if compression[0] is not None else None
    chunks_mc_infos = (32,) + mc_infos.shape[1:] if compression[0] is not None else None
    fletcher32 = True if compression[0] is not None else False

    dset_hists = h5f.create_dataset('x', data=hists, dtype='uint8', fletcher32=fletcher32, chunks=chunks_hists,
                                    compression=compression[0], compression_opts=compression[1], shuffle=False)
    dset_mc_infos = h5f.create_dataset('y', data=mc_infos, dtype='float32', fletcher32=fletcher32, chunks=chunks_mc_infos,
                                       compression=compression[0], compression_opts=compression[1], shuffle=False)

    h5f.close()

if make_channel_statistics==True:
    make_channel_statistics(file)
else:
    generate_file(original_train_file, outfile_train, fraction, sum_over_axis, reshape_to_channel_and_shuffle, only_doms_with_more_then)
    generate_file(original_test_file,  outfile_test,  fraction, sum_over_axis, reshape_to_channel_and_shuffle, only_doms_with_more_then)
    print("Done.")
