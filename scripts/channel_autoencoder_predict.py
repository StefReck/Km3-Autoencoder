# -*- coding: utf-8 -*-
"""
Load a channel id autoencoder model and predict on some train files.
"""

from keras.models import load_model
import h5py
import numpy as np
import argparse

from get_dataset_info import get_dataset_info

def parse_input():
    parser = argparse.ArgumentParser(description='Predict on channel data')
    parser.add_argument('model_name', type=str)

    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
model_name = params["model_name"]


model=load_model(model_name)

dataset_info_dict=get_dataset_info("xyzc_flat")
test_file = dataset_info_dict["test_file"]
f = h5py.File(test_file, "r")

batch=f["x"][:10]
pred=model.predict_on_batch(batch)

for i in range(len(batch)):
    print("Original")
    print(batch[i])
    print("Prediction")
    print(pred[i])
    print("\n")

