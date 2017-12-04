# -*- coding: utf-8 -*-

import h5py
import numpy as np

#Datei die alles enthaelt
file=h5py.File("Daten\example.h5", "r")

#alle attribute der datei
for item in file.attrs.keys():
    print(item + ":", file.attrs[item])
    
# Alle datasets die die datei enthaelt
print("\n Datasets:")
for dset_name in file.keys():
    print(dset_name)

#ein spezielles dataset
mctracks = file['mc_tracks']
print("\n Length of mctracks:", len(mctracks))
#mctracks hat 25228 reihen (events)

#jede zeile hat 15 eintraege, z.B. bjorkeny oder dir_x:
print("\n Attributes of mc_tracks:")
for attr in mctracks.attrs.keys():
    print(attr + ":", mctracks.attrs[attr])
    
#@@@@@@@@@@@@@@@@@@@ Warum geht nicht mc_tracks.bjorkeny so wie l 20 filetohits bzw file['mc_tracks'].bjorkeny
bjorken_y_array=mctracks["bjorkeny"]
print("\n Wo ist bjorkeny null? --> primaeres event:")
p = np.where(bjorken_y_array != 0.0)[0][0]
print(p)
primary_event=mctracks[p]


#ein anderes spezielles dataset, hits ist eine group
hits=file['hits']
print("\n Length of hits:", len(hits))
#hits hat 16 member, und zwar die folgenden:
print("\n Members of hits:")
for member in hits.keys():
    print(member)
print("\n")
#channel id ist id des PM:
channel_id = np.array(hits["channel_id"])
chan, chan_count = np.unique(channel_id, return_counts=True)
for i in range(len(chan)):
    print("Channel",chan[i],"appears",chan_count[i],"times")

#pos_x hat 7270909 eintraege
print("\n Entries in pos_x: ",len(hits["pos_x"]))

#es stehen da also 7270909 events drin, abgkrzt 3:
pos_x = np.array([1,2,3])
pos_y = np.array([1,2,3])
pos_z = np.array([1,2,3])
time =  np.array([1,2,3])

#man will jetzt jeweils x,y,z,t eines events in einem array haben, also [1,1,1,1],[2,2,2,2],[3,3,3,3]
#das geht so:
ax_ex = np.newaxis
event_hits_ex = np.concatenate([pos_x[:, ax_ex], pos_y[:, ax_ex], pos_z[:, ax_ex], time[:, ax_ex]], axis=1)
print("\n Concatenated events:", event_hits_ex)


#simulierte daten:
#@@@@@@@@@@@@@@@@@@@ Warum geht nicht hits.pos_x so wie l 69 filetohits
#@@@@@@@@@@@@@@@@@@@ Warum brauch ich hier np.array?
pos_x = np.array(hits["pos_x"]).astype('float32')
pos_y = np.array(hits["pos_y"]).astype('float32')
pos_z = np.array(hits["pos_z"]).astype('float32')
time = np.array(hits["time"]).astype('float32')
#in einer datei sind viele Messungen (ca. 3500) gespeichert, identifizierbar mittels id
#auch event_id hat länge 7270909
event_id=np.array(hits["event_id"])

ax = np.newaxis
#Dimension: 7270909 x 4
event_hits = np.concatenate([pos_x[:, ax], pos_y[:, ax], pos_z[:, ax], time[:, ax]], axis=1)





# Ab hier reinkopiert vom hist git:

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

def calculate_bin_edges(n_bins, fname_geo_limits):
    """
    Calculates the bin edges for the later np.histogramdd actions based on the number of specified bins. 
    This is performed in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.
    :param tuple n_bins: contains the desired number of bins for each dimension. [n_bins_x, n_bins_y, n_bins_z]
    :param str fname_geo_limits: filepath of the .txt ORCA geometry file.
    :return: ndarray(ndim=1) x_bin_edges, y_bin_edges, z_bin_edges: contains the resulting bin edges for each dimension.
    """
    #print ("Reading detector geometry in order to calculate the detector dimensions from file " + fname_geo_limits)
    geo = np.loadtxt(fname_geo_limits)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis = 0), np.nanmax(geo, axis = 0)
    #print ('Detector dimensions [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]: ' + str(geo_limits))

    x_bin_edges = np.linspace(geo_limits[0][1] - 9.95, geo_limits[1][1] + 9.95, num=n_bins[0] + 1) #try to get the lines in the bin center 9.95*2 = average x-separation of two lines
    y_bin_edges = np.linspace(geo_limits[0][2] - 9.75, geo_limits[1][2] + 9.75, num=n_bins[1] + 1) # Delta y = 19.483
    z_bin_edges = np.linspace(geo_limits[0][3] - 4.665, geo_limits[1][3] + 4.665, num=n_bins[2] + 1) # Delta z = 9.329

    #calculate_bin_edges_test(geo, y_bin_edges, z_bin_edges) # test disabled by default. Activate it, if you change the offsets in x/y/z-bin-edges

    return x_bin_edges, y_bin_edges, z_bin_edges

def compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_3d_hists):
    x = event_hits[:, 0:1]
    y = event_hits[:, 1:2]
    z = event_hits[:, 2:3]
    t = event_hits[:, 3:4]
    t_start, t_end = get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15)
    hist_xyz = np.histogramdd(event_hits[:, 0:3], bins=(x_bin_edges, y_bin_edges, z_bin_edges), range=((min(x_bin_edges), max(x_bin_edges)), (min(y_bin_edges), max(y_bin_edges)), (min(z_bin_edges), max(z_bin_edges)) ))
    hist_xyt = np.histogramdd(np.concatenate([x, y, t], axis=1), bins=(x_bin_edges, y_bin_edges, n_bins[3]), range=((min(x_bin_edges), max(x_bin_edges)), (min(y_bin_edges), max(y_bin_edges)), (t_start, t_end)))
    return np.array(hist_xyz[0], dtype=np.uint8),np.array(hist_xyt[0], dtype=np.uint8)


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
    #Dann ist t ein N x 1 array, würde man [:,3] schrieben, wäre es ein N array
    
    t_start, t_end = get_time_parameters(t, t_start_margin=0.15, t_end_margin=0.15)

    hist_xyzt = np.histogramdd(event_hits[:, 0:4], bins=(x_bin_edges, y_bin_edges, z_bin_edges, n_bins[3]),
                               range=((min(x_bin_edges),max(x_bin_edges)),(min(y_bin_edges),max(y_bin_edges)),
                                      (min(z_bin_edges),max(z_bin_edges)),(t_start, t_end)))

    #all_4d_to_4d_hists.append(np.array(hist_xyzt[0], dtype=np.uint8))
    return [np.array(hist_xyzt[0], dtype=np.uint8)]

def make_4d_test_hist():
    filename_geo_limits = 'ORCA_Geo_115lines.txt'
    n_bins=(11,13,18,50)
    x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges( n_bins, filename_geo_limits)
    all_4d_to_4d_hists = []
    all_4d_to_4d_hists = compute_4d_to_4d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, all_4d_to_4d_hists)
    np.save("test_hist_4d", all_4d_to_4d_hists)
    
def make_3d_test_hist():
    filename_geo_limits = 'ORCA_Geo_115lines.txt'
    n_bins=(11,13,18,50)
    x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges( n_bins, filename_geo_limits)
    hist=[]
    hist = compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, hist)
    np.save("test_hist_3d", hist[0])
    np.save("test_hist_3d_xyt", hist[1])

#make_3d_test_hist()
#make_4d_test_hist()
    
    
    
    
    
    
    
    