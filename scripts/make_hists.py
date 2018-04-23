# -*- coding: utf-8 -*-

import h5py
import numpy as np
import km3pipe as kp


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ rauskopiert:

def get_primary_track_index(event_blob):
    """
    Gets the index of the primary (neutrino) track.
    Uses bjorkeny in order to get the primary track, since bjorkeny!=0 for the initial interacting neutrino.
    :param kp.io.HDF5Pump.blob event_blob: HDF5Pump event blob.
    :return: int primary index: Index of the primary track (=neutrino) in the 'McTracks' branch.
    """
    bjorken_y_array = event_blob['bjorkeny']
    primary_index = np.where(bjorken_y_array != 0.0)[0][0]
    return primary_index

def get_event_data(event_blob, geo):
    p = get_primary_track_index(event_blob)

    # parse tracks [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    event_id = event_blob['EventInfo'].event_id[0]
    particle_type = event_blob['McTracks'][p].type
    energy = event_blob['McTracks'][p].energy
    is_cc = event_blob['McTracks'][p].is_cc
    bjorkeny = event_blob['McTracks'][p].bjorkeny
    dir_x = event_blob['McTracks'][p].dir[0]
    dir_y = event_blob['McTracks'][p].dir[1]
    dir_z = event_blob['McTracks'][p].dir[2]
    time = event_blob['McTracks'][p].time

    event_track = np.array([event_id, particle_type, energy, is_cc, bjorkeny, dir_x, dir_y, dir_z, time], dtype=np.float32)

    # parse hits [x,y,z,time]
    #Veranderter code:
    pos_x = np.array(hits["pos_x"]).astype('float32')
    pos_y = np.array(hits["pos_y"]).astype('float32')
    pos_z = np.array(hits["pos_z"]).astype('float32')
    time = np.array(hits["time"]).astype('float32')

    ax = np.newaxis
    event_hits = np.concatenate([pos_x[:, ax], pos_y[:, ax], pos_z[:, ax], time[:, ax]], axis=1)

    # event_hits: 2D hits array for one event, event_track: 1D track array containing event information
    return event_hits, event_track


def get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15):
    """
    Gets the fundamental time parameters in one place for cutting a time residual.
    Later on these parameters cut out a certain time span of events specified by t_start and t_end.
    :param ndarray(ndim=1) t: time column of the event_hits array.
    :param float t_start_margin: defines the start time of the selected timespan with t_mean - t_start * t_diff.
    :param float t_end_margin: defines the end time of the selected timespan with t_mean + t_start * t_diff.
    :return: float t_start, t_end: absolute start and end time that will be used for the later timespan cut.
                                   Events in this timespan are accepted, others are rejected.
    """
    t_min = np.amin(t)
    t_max = np.amax(t)
    t_diff = t_max - t_min
    t_mean = t_min + 0.5 * t_diff

    t_start = t_mean - t_start_margin * t_diff
    t_end = t_mean + t_end_margin * t_diff

    return t_start, t_end
    #return t_min, t_max # for no time cut

def calculate_bin_edges(n_bins, fname_geo_limits):
    """
    Calculates the bin edges for the later np.histogramdd actions based on the number of specified bins. 
    This is performed in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.
    :param tuple n_bins: contains the desired number of bins for each dimension. [n_bins_x, n_bins_y, n_bins_z]
    :param str fname_geo_limits: filepath of the .txt ORCA geometry file.
    :return: ndarray(ndim=1) x_bin_edges, y_bin_edges, z_bin_edges: contains the resulting bin edges for each dimension.
    """
    #print "Reading detector geometry in order to calculate the detector dimensions from file " + fname_geo_limits
    geo = np.loadtxt(fname_geo_limits)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis = 0), np.nanmax(geo, axis = 0)
    #print 'Detector dimensions [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]: ' + str(geo_limits)

    x_bin_edges = np.linspace(geo_limits[0][1] - 9.95, geo_limits[1][1] + 9.95, num=n_bins[0] + 1) #try to get the lines in the bin center 9.95*2 = average x-separation of two lines
    y_bin_edges = np.linspace(geo_limits[0][2] - 9.75, geo_limits[1][2] + 9.75, num=n_bins[1] + 1) # Delta y = 19.483
    z_bin_edges = np.linspace(geo_limits[0][3] - 4.665, geo_limits[1][3] + 4.665, num=n_bins[2] + 1) # Delta z = 9.329

    #calculate_bin_edges_test(geo, y_bin_edges, z_bin_edges) # test disabled by default. Activate it, if you change the offsets in x/y/z-bin-edges

    return x_bin_edges, y_bin_edges, z_bin_edges

def compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_2d_hists): #, event_track, do2d_pdf):
    """
    Computes 2D numpy histogram 'images' from the 4D data.
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyzt) data for a certain eventID. [positions_xyz, time]
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction.
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction.
    :param tuple n_bins: Contains the number of bins that should be used for each dimension (x,y,z,t).
    :param list all_4d_to_2d_hists: contains all 2D histogram projections.
    :param ndarray(ndim=2) event_track: contains the relevant mc_track info for the event in order to get a nice title for the pdf histos.
    :param bool do2d_pdf: if True, generate 2D matplotlib pdf histograms.
    :return: appends the 2D histograms to the all_4d_to_2d_hists list.
    """
    x = event_hits[:, 0]
    y = event_hits[:, 1]
    z = event_hits[:, 2]
    t = event_hits[:, 3]

    # analyze time
    t_start, t_end = get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15)

    # create histograms for this event
    hist_xy = np.histogram2d(x, y, bins=(x_bin_edges, y_bin_edges))  # hist[0] = H, hist[1] = xedges, hist[2] = yedges
    hist_xz = np.histogram2d(x, z, bins=(x_bin_edges, z_bin_edges))
    hist_yz = np.histogram2d(y, z, bins=(y_bin_edges, z_bin_edges))

    hist_xt = np.histogram2d(x, t, bins=(x_bin_edges, n_bins[3]), range=((min(x_bin_edges), max(x_bin_edges)), (t_start, t_end)))
    hist_yt = np.histogram2d(y, t, bins=(y_bin_edges, n_bins[3]), range=((min(y_bin_edges), max(y_bin_edges)), (t_start, t_end)))
    hist_zt = np.histogram2d(z, t, bins=(z_bin_edges, n_bins[3]), range=((min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))

    # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
    all_4d_to_2d_hists.append((np.array(hist_xy[0], dtype=np.uint8),
                               np.array(hist_xz[0], dtype=np.uint8),
                               np.array(hist_yz[0], dtype=np.uint8),
                               np.array(hist_xt[0], dtype=np.uint8),
                               np.array(hist_yt[0], dtype=np.uint8),
                               np.array(hist_zt[0], dtype=np.uint8)))


def compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_3d_hists):
    """
    Computes 3D numpy histogram 'images' from the 4D data.
    Careful: Currently, appending to all_4d_to_3d_hists takes quite a lot of memory (about 200MB for 3500 events).
    In the future, the list should be changed to a numpy ndarray.
    (Which unfortunately would make the code less readable, since an array is needed for each projection...)
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyzt) data for a certain eventID. [positions_xyz, time]
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction. 
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction.
    :param tuple n_bins: Declares the number of bins that should be used for each dimension (x,y,z,t).
    :param list all_4d_to_3d_hists: contains all 3D histogram projections.
    :return: appends the 3D histograms to the all_4d_to_3d_hists list. [xyz, xyt, xzt, yzt, rzt]
    """
    x = event_hits[:, 0:1]
    y = event_hits[:, 1:2]
    z = event_hits[:, 2:3]
    t = event_hits[:, 3:4]

    t_start, t_end = get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15)

    #New:
    #condition to filter for in xyz histogram
    con=(t>t_start) & (t<t_end)
    
    
    hist_xyz = np.histogramdd(event_hits[con[:,0], 0:3], bins=(x_bin_edges, y_bin_edges, z_bin_edges))

    hist_xyt = np.histogramdd(np.concatenate([x, y, t], axis=1), bins=(x_bin_edges, y_bin_edges, n_bins[3]),
                              range=((min(x_bin_edges), max(x_bin_edges)), (min(y_bin_edges), max(y_bin_edges)), (t_start, t_end)))
    hist_xzt = np.histogramdd(np.concatenate([x, z, t], axis=1), bins=(x_bin_edges, z_bin_edges, n_bins[3]),
                              range=((min(x_bin_edges), max(x_bin_edges)), (min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))
    hist_yzt = np.histogramdd(event_hits[:, 1:4], bins=(y_bin_edges, z_bin_edges, n_bins[3]),
                              range=((min(y_bin_edges), max(y_bin_edges)), (min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))

    # add a rotation-symmetric 3d hist
    """
    r = np.sqrt(x * x + y * y)
    rzt = np.concatenate([r, z, t], axis=1)
    hist_rzt = np.histogramdd(rzt, bins=(n_bins[0], n_bins[2], n_bins[3]), range=((np.amin(r), np.amax(r)), (np.amin(z), np.amax(z)), (t_start, t_end)))
    """
    
    all_4d_to_3d_hists.append((np.array(hist_xyz[0], dtype=np.uint8),
                               np.array(hist_xyt[0], dtype=np.uint8),
                               np.array(hist_xzt[0], dtype=np.uint8),
                               np.array(hist_yzt[0], dtype=np.uint8)))#, np.array(hist_rzt[0], dtype=np.uint8)))
    
def compute_4d_to_4d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_4d_hists):
    """
    Computes 4D numpy histogram 'images' from the 4D data.
    :param ndarray(ndim=2) event_hits: 2D array that contains the hits (_xyzt) data for a certain eventID. [positions_xyz, time]
    :param ndarray(ndim=1) x_bin_edges: bin edges for the X-direction.
    :param ndarray(ndim=1) y_bin_edges: bin edges for the Y-direction.
    :param ndarray(ndim=1) z_bin_edges: bin edges for the Z-direction.
    :param tuple n_bins: Declares the number of bins that should be used for each dimension (x,y,z,t).
    :param list all_4d_to_4d_hists: contains all 4D histogram projections.
    :return: appends the 4D histogram to the all_4d_to_4d_hists list. [xyzt]
    """
    t = event_hits[:, 3:4]
    t_start, t_end = get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15)

    hist_xyzt = np.histogramdd(event_hits[:, 0:4], bins=(x_bin_edges, y_bin_edges, z_bin_edges, n_bins[3]),
                               range=((min(x_bin_edges),max(x_bin_edges)),(min(y_bin_edges),max(y_bin_edges)),
                                      (min(z_bin_edges),max(z_bin_edges)),(t_start, t_end)))

    all_4d_to_4d_hists.append(np.array(hist_xyzt[0], dtype=np.uint8))
    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Ende rauskopiert



file=h5py.File("Daten\example.h5", "r")
hits=file['hits']
pos_x = np.array(hits["pos_x"]).astype('float32')
pos_y = np.array(hits["pos_y"]).astype('float32')
pos_z = np.array(hits["pos_z"]).astype('float32')
time = np.array(hits["time"]).astype('float32')
event_id=np.array(hits["event_id"])

#Filter for events with high energy:
high_e=file['mc_tracks']["energy"] > 99
high_e_id=file['mc_tracks']["event_id"][high_e]


#Filter out a specific event from the 3500ish in the same file
target_event= high_e_id[0]
pos_x=pos_x[event_id==target_event]
pos_y=pos_y[event_id==target_event]
pos_z=pos_z[event_id==target_event]
time=time[event_id==target_event]

ax = np.newaxis
#Dimension: 2120 x 4:   x,y,z,t
event_hits = np.concatenate([pos_x[:, ax], pos_y[:, ax], pos_z[:, ax], time[:, ax]], axis=1)


def compute_histograms():
    filename_geo_limits = 'Daten\ORCA_Geo_115lines.txt'
    n_bins=(11,13,18,50)
    
    x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges( n_bins, filename_geo_limits)

    all_4d_to_2d_hists = []
    compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_2d_hists)

    #4 diagramme im array, jeweils 3d matrizen
    all_4d_to_3d_hists = []
    compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_3d_hists)

    #1 diagramm im array, 4d matrix
    all_4d_to_4d_hists = []
    compute_4d_to_4d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_4d_hists)

    return all_4d_to_2d_hists, all_4d_to_3d_hists, all_4d_to_4d_hists


#all_4d_to_2d_hists, all_4d_to_3d_hists, all_4d_to_4d_hists = compute_histograms()

#np.save("Daten/test_hist_2d", all_4d_to_2d_hists)
#np.save("Daten/test_hist_3d", all_4d_to_3d_hists)
#np.save("Daten/test_hist_4d", all_4d_to_4d_hists)












def store_histograms_as_hdf5(hists, filepath_output):

    h5f = h5py.File(filepath_output, 'w')
    dset_hists = h5f.create_dataset('x', data=hists, dtype='uint8')
    #dset_mc_infos = h5f.create_dataset('y', data=mc_infos, dtype='float32')
    h5f.close()

def main(n_bins, do2d=True, do2d_pdf=(False, 10), do3d=True, do4d=False, do_mc_hits=False, use_calibrated_file=True, data_cuts=None):
    """
    Main code. Reads raw .hdf5 files and creates 2D/3D histogram projections that can be used for a CNN
    :param tuple(int) n_bins: Declares the number of bins that should be used for each dimension (x,y,z,t).
    :param bool do2d: Declares if 2D histograms should be created.
    :param (bool, int) do2d_pdf: Declares if pdf visualizations of the 2D histograms should be created. Cannot be called if do2d=False.
                                 The event loop will be stopped after the integer specified in the second argument.
    :param bool do3d: Declares if 3D histograms should be created.
    :param bool do4d: Declares if 4D histograms should be created.
    :param bool do_mc_hits: Declares if hits (False, mc_hits + BG) or mc_hits (True) should be processed
    :param bool use_calibrated_file: Declares if the input file is already calibrated (pos_x/y/z, time) or not.
    :param dict data_cuts: Dictionary that contains information about any possible cuts that should be applied.
                           Supports the following cuts: 'triggered', 'energy_lower_limit'
    """
    if data_cuts is None: data_cuts={'triggered': False, 'energy_lower_limit': 0}

    filename_geo_limits = 'Daten/ORCA_Geo_115lines.txt' # used for calculating the dimensions of the ORCA can

    x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges(n_bins, filename_geo_limits)

    all_4d_to_2d_hists, all_4d_to_3d_hists, all_4d_to_4d_hists = [], [], []
    mc_infos = []

    # Initialize HDF5Pump of the input file
    event_pump = kp.io.hdf5.HDF5Pump(filename="Daten\example.h5")
    #print "Generating histograms from the hits in XYZT format for files based on " + filename_input
    for i, event_blob in enumerate(event_pump):
        print(i,event_blob)
        if i % 10 == 0:
            print ('Event No. ' + str(i))

        # filter out all hit and track information belonging that to this event
        geo=None
        event_hits, event_track = get_event_data(event_blob, geo)

        if event_track[2] < data_cuts['energy_lower_limit']: # Cutting events with energy < threshold (default=0)
            #print 'Cut an event with an energy of ' + str(event_track[2]) + ' GeV'
            continue

        # event_track: [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
        mc_infos.append(event_track)

        if do2d:
            compute_4d_to_2d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_2d_hists, event_track, do2d_pdf[0])

        if do3d:
            compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_3d_hists)

        if do4d:
            compute_4d_to_4d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, all_4d_to_4d_hists)

        if do2d_pdf[0] is True:
            if i >= do2d_pdf[1]:
                glob.pdf_2d_plots.close()
                break

    if do3d:
        store_histograms_as_hdf5(np.stack([hist_tuple[0] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Daten/xyz.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[1] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Daten/xyt.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[2] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Daten/xzt.h5')
        store_histograms_as_hdf5(np.stack([hist_tuple[3] for hist_tuple in all_4d_to_3d_hists]), np.array(mc_infos), 'Daten/yzt.h5')

main(n_bins=(11,13,18,50))







