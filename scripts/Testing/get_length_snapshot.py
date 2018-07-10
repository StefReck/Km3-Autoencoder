# -*- coding: utf-8 -*-
"""
Output length of snapshots.
"""

import numpy as np
import h5py
file="JTE.KM3Sim.gseagen.elec-CC.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016.998.h5"
file="JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.99.h5"
f=h5py.File(file, "r")
hits=f["hits"]
event_id_index=7
time_index=-3
min_event_id = np.min(hits["event_id"])
max_event_id = np.max(hits["event_id"])
length = []
for event_id in range(min_event_id, max_event_id+1):
    if event_id%100==0:
        print(event_id,"/",max_event_id)
    hits_for_this_event = np.array(hits["event_id"])==event_id
    time_hits=hits["time"][hits_for_this_event]
    length.append(np.max(time_hits)-np.min(time_hits))
    
print(length)
#elec CC:   3196+-183 ns (std)
#muon CC:   3224+-196 ns (std)


