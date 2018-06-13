import h5py
import numpy as np

# x, y, z, t, c
#11,13,18,50,31

mode="broken15"

#original_train_file="/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/xyz_channel_-350+850/concatenated/elec-CC_and_muon-CC_xyzc_train_1_to_480_shuffled_0.h5"
    
#outfile_train="channel/elec-CC_and_muon-CC_c_train_1_to_240_shuffled_0.h5"
#outfile_test= "channel/elec-CC_and_muon-CC_c_test_481_to_540_shuffled_0.h5"

#Default values:
#percentage of events to keep
fraction=1
#which axis including filesize to sum over, None if no sum
#e.g. X,11,13,18,50 --> X,11,18,50 axis=2
sum_over_axis=None
reshape_to_channel_and_shuffle=False
only_doms_with_more_then=0
broken_mode=None


if mode=="channel_event":
    #
    
    original_train_file="channel/elec-CC_and_muon-CC_c_event_train_1_to_240_shuffled_0.h5"
    original_test_file ="/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/h5_input_projections_3-100GeV/4dTo4d/xyz_channel_-350+850/concatenated/elec-CC_and_muon-CC_xyzc_test_481_to_600_shuffled_0.h5"
    
    outfile_train="channel/elec-CC_and_muon-CC_c_event_train_1_to_240_shuffled_0.h5"
    outfile_test ="channel/elec-CC_and_muon-CC_c_event_test_481_to_540_shuffled_0.h5"
       
    #11,13,18,31
    
    #percentage of events to keep
    fraction=1
    #which axis including filesize to sum over, None if no sum
    #e.g. X,11,13,18,50 --> X,11,18,50 axis=2
    sum_over_axis=None

    reshape_to_channel_and_shuffle=True
    only_doms_with_more_then=2

elif mode=="channel_up_manip":
    #all channel ids that are looking upward have reduced counts (expectat.)
    #Input: xztc (11,18,50,31)
    #Output: xzt (11,18,50)
    original_train_file="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xztc/elec-CC_and_muon-CC_xyzt_train_1_to_240_shuffled_0.h5"
    original_test_file ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xztc/elec-CC_and_muon-CC_xyzt_test_481_to_600_shuffled_0.h5"
    
    outfile_train="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken5/elec-CC_and_muon-CC_xzt_broken5_event_train_1_to_240_shuffled_0.h5"
    outfile_test ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken5/elec-CC_and_muon-CC_xzt_broken5_test_481_to_600_shuffled_0.h5"
       
    #percentage of events to keep
    fraction=1
    #which axis including filesize to sum over, None if no sum
    #e.g. X,11,13,18,50 --> X,11,18,50 axis=2
    sum_over_axis=4

    reshape_to_channel_and_shuffle=False
    only_doms_with_more_then=0
    broken_mode=5

elif mode=="broken12":
    #Add noise corr to Energy
    #Input: xzt (11,18,50)
    #Output: xzt (11,18,50)
    original_train_file="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
    original_test_file ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    
    outfile_train="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken12/train_muon-CC_and_elec-CC_each_240_xzt_broken12_shuffled.h5"
    outfile_test ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken12/test_muon-CC_and_elec-CC_each_60_xzt_broken_12_shuffled.h5"
       
    #percentage of events to keep
    fraction=1
    broken_mode=12
    
elif mode=="broken13":
    #Add noise corr to Energy
    #Input: xzt (11,18,50)
    #Output: xzt (11,18,50)
    original_train_file="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
    original_test_file ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    
    outfile_train="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken13/train_muon-CC_and_elec-CC_each_240_xzt_broken13_shuffled.h5"
    outfile_test ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken13/test_muon-CC_and_elec-CC_each_60_xzt_broken_13_shuffled.h5"
       
    #percentage of events to keep
    fraction=1
    broken_mode=13
    
elif mode=="broken14":
    #Add noise corr to Energy
    #Input: xzt (11,18,50)
    #Output: xzt (11,18,50)
    original_train_file="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
    original_test_file ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    
    outfile_train="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken14/train_muon-CC_and_elec-CC_each_240_xzt_broken14_shuffled.h5"
    outfile_test ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken14/test_muon-CC_and_elec-CC_each_60_xzt_broken_14_shuffled.h5"
    
    #percentage of events to keep
    fraction=1
    broken_mode=14
    
elif mode=="broken15":
    #Reduce Quantum efficiency of doms according to x-Coordinate
    #can be caused by lines having been in the water for different time spans
    #Input: xzt (11,18,50)
    #Output: xzt (11,18,50)
    original_train_file="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/train_muon-CC_and_elec-CC_each_240_xzt_shuffled.h5"
    original_test_file ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt/test_muon-CC_and_elec-CC_each_60_xzt_shuffled.h5"
    
    outfile_train="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken15/train_muon-CC_and_elec-CC_each_240_xzt_broken15_shuffled.h5"
    outfile_test ="/home/woody/capn/mppi013h/Km3-Autoencoder/data/xzt_broken15/test_muon-CC_and_elec-CC_each_60_xzt_broken_15_shuffled.h5"
    
    #percentage of events to keep
    fraction=1
    broken_mode=15

#show how often certain total number of hits in a dom occur; No files will be generated
make_statistics = False
#save it as a npy file
save_statistics_name = original_train_file.split("/")[-1][:-3]+"_statistics_fraction_"+str(fraction)+".npy"


def make_broken5_manip(hists_temp, sum_channel = True):
    #Input (X,11,18,50,31) xztc hists
    #Output: (X,11,18,50) manipulated xzt hists
    
    #all doms facing upwards AND having >0 counts get reduced by a
    #binomial distribution with n=2,p=0.4 (but not below 0!)
    
    #1=upwards facing, taken from the paper
    up_mask=np.array([True,]*12 + [False,]*19) 
    #chance for upward facing doms with >0 counts to have one count removed:
    #chance
        
    #the counts to subtract: upwards facing doms have a chance% chance of getting one count removed
    #subt = np.random.choice([0,1,2],size=hists_temp.shape, p=[0.3,0.5,0.2])
    subt = np.random.binomial(2, 0.4, size=hists_temp.shape)
    subt = np.multiply(up_mask, subt)
    #subtract
    hists_temp=hists_temp-subt
    #negative counts are not allowed, they are set to 0 instead
    hists_temp = np.clip(hists_temp, 0, None)
    
    #sum over channel axis to get X,11,18,50 xzt data
    if sum_channel == True:
        hists_temp=np.sum(hists_temp, axis=-1)
    
    return hists_temp

def add_energy_correlated_noise(xs, true_energies, broken_mode=12):
    """Adds additional poisson noise whose expectation value
    is proportional to some function of energy"""
    expectation_value_10_kHz=0.08866 # for 10kHz noise
    if broken_mode==12:
        #Expectation value linearly decreasing from 10 kHz at 3 GeV to 0 kHz at 100 GeV
        poisson_noise_expectation_value = expectation_value_10_kHz * (100-true_energies)/97
    elif broken_mode==13:
        #Expectation value linearly increasing from 0 kHz at 3 GeV to 5 kHz at 100 GeV
        poisson_noise_expectation_value = 0.5 * expectation_value_10_kHz * (1-(100-true_energies)/97)
    elif broken_mode==14:
        #Same as 13, but maximum expectation value is 2 kHz
        poisson_noise_expectation_value = 0.2 * expectation_value_10_kHz * (1-(100-true_energies)/97) 
        
    #noise has the shape (dims, batchsize), while xs has the shape (batchsize, dims)
    noise = np.random.poisson(poisson_noise_expectation_value, size=xs.shape[1:]+true_energies.shape)
    #permute so that noise has shape (batchsize, dims), just as xs
    noise = np.transpose(noise, np.roll(np.arange(len(noise.shape)), shift=1) )
    
    xs = xs + noise
    return xs

def get_broken15_efficiency(make_plot=False):
    #expectation value of the quantum efficiency, 95% (or so) at x=0, 60% at x=11
    #Will have shape 11,18,50
    quantum_efficiency = np.ones((11,18,50))
    for x_no in range(11):
        quantum_efficiency[x_no,:,:] *= (1 - (0.4 * (x_no+1)/11))
    #now scale by some random value, gaussian distr.
    #this random value is the same for all time bins, aka fix in time for every DOM
    np.random.seed(64)
    random_value = np.random.normal(1,0.1,size=(11,18,1))
    random_value = np.repeat(random_value, 50, axis=-1)
    quantum_efficiency = np.clip(quantum_efficiency * random_value, 
                                 a_min=0, a_max=1)
    #this quantum_efficiency array is the same for all data!
    if make_plot:
        import matplotlib.pyplot as plt
        figsize = [5,5.5]   
        font_size=14
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': font_size})
        #plt.grid()
        image = quantum_efficiency[:,:,0] #11,18
        plt.imshow(image.T)
        
        #plt.hist(quantum_efficiency.reshape((quantum_efficiency.shape[0], quantum_efficiency.shape[1]*quantum_efficiency.shape[2])).T, 
        #                    bins=15, histtype="barstacked")
        plt.xlabel("x Bin no")
        plt.xticks(np.arange(0,11,2), np.arange(1,12,2))
        plt.ylabel("z Bin no")
        plt.yticks((0,6,12,17), (1,7,13,18))
        cbar = plt.colorbar()
        cbar.set_label('Efficiency')
        #usually saved as broken_energy/broken15_efficiency_plot.pdf
        plt.show()
    return quantum_efficiency
#xs=get_broken15_efficiency(True)
#raise
#xs_org = np.random.randint(0,10,(1,11,18,50)).astype(int)
    
def make_broken15_manip(xs, quantum_efficiency):
    """ Reduce the Count number based on the x-Coordinate of the string.
    xs.shape=(bs,11,18,50)"""
    #n counts get reduced by binomial distribution with n tries and p=1-quantum_efficiency
    xs=xs.astype(int)
    xs = xs - np.random.binomial(xs, 1-quantum_efficiency)
    return xs.astype(int)


def generate_file(file, save_to, fraction, sum_over_axis, reshape_to_channel_and_shuffle, only_doms_with_more_then, broken_mode=None):     
    print("Generating file", save_to)
    f=h5py.File(file, "r")
    shape=f["x"].shape #e.g. (X,11,13,18,50)
    print("Original shape of hists:", shape)
    up_to_which=int(shape[0]*fraction)
    print("Events left after cut:", up_to_which)

    if broken_mode==5:
        #do the task in several steps
        how_many_steps=500
        per_step=int(up_to_which/how_many_steps)
        print("Taking", how_many_steps,"steps with", per_step,"events each.")
        print("New file will have shape",(per_step*how_many_steps,)+f["x"].shape[1:-1])
        
        hists = np.zeros(shape=(per_step*how_many_steps,)+f["x"].shape[1:-1]) #e.g. 11,18,50
        
        for step in range(how_many_steps):
            print("Step", step, "/", how_many_steps)
            part_of_datafile = (per_step*step, per_step*(step+1))
        
            hists_temp=f["x"][part_of_datafile[0]:part_of_datafile[1]] #X,11,18,50,31
            hists[part_of_datafile[0]:part_of_datafile[1]]=make_broken5_manip(hists_temp)
    elif broken_mode==12 or broken_mode==13 or broken_mode==14:
        hists=f["x"][:up_to_which]
        true_energies = f["y"][:up_to_which][:,2]
        hists = add_energy_correlated_noise(hists, true_energies, broken_mode)
            
    elif broken_mode==15:
        hists=f["x"][:up_to_which]
        efficency = get_broken15_efficiency()
        np.random.seed()
        hists = make_broken15_manip(hists, efficency)
        
    else:
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
            np.random.shuffle(hists)
        
        mc_infos=np.zeros(100)
    else:
        mc_infos=f["y"][:up_to_which]
        
    print("New shape (hists):", hists.shape, ", mc infos:", mc_infos.shape)
    store_histograms_as_hdf5(hists, mc_infos, save_to, compression=("gzip", 1))


def make_channel_statistics(file, fraction, save_statistics_name):
    f=h5py.File(file, "r")
    shape=f["x"].shape #e.g. (X,11,13,18,50)
    print("Original shape of hists:", shape)
    up_to_which=int(shape[0]*fraction)
    hists=f["x"][:up_to_which]
    #number of occurances of total hits in a whole dom
    dom_hits = np.sum(hists, axis=-1).flatten()
    count_number = np.bincount(dom_hits.astype(int))
    print("How many DOMs are there with how many total hits:")
    for no_counts in range(len(count_number)):
        print(no_counts, "Hits:\t", count_number[no_counts], "times")
    if save_statistics_name != None:
        np.save(save_statistics_name, count_number)
        

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

if make_statistics==True:
    make_channel_statistics(original_train_file, fraction, save_statistics_name)
else:
    generate_file(original_train_file, outfile_train, fraction, sum_over_axis, reshape_to_channel_and_shuffle, only_doms_with_more_then, broken_mode)
    generate_file(original_test_file,  outfile_test,  fraction, sum_over_axis, reshape_to_channel_and_shuffle, only_doms_with_more_then, broken_mode)
    print("Done.")
