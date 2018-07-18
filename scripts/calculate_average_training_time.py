# -*- coding: utf-8 -*-
"""
Calculate average training time.
"""

import argparse
def parse_input():
    parser = argparse.ArgumentParser(description='Calculate average training time.')
    parser.add_argument('test_file', type=str, help='Test file of the training of a model.')

    return parser.parse_args()
params = parse_input()

from plotting.plot_statistics import make_dicts_from_files

test_file = params.test_file

def convert_to_minutes(time_string):
    #input: e.g. 09:37:23_to_10:00:56
    start_time, __, end_time = time_string.split("_")
    
    hours, minutes, secs = [int(e) for e in start_time.split(":")]
    ehours, eminutes, esecs = [int(e) for e in end_time.split(":")]
    
    time_diff_minutes = (ehours - hours)*60 + (eminutes - minutes)%60 - (minutes>eminutes)*60
    
    return time_diff_minutes
    
def get_mean_test_time(test_file):
    test_file_data = make_dicts_from_files([test_file,])
    test_times = test_file_data[0]["Time"]
    training_times = [convert_to_minutes(test_time) for test_time in test_times]
    return float(sum(training_times))/len(training_times)

mean_test_time = get_mean_test_time(test_file)
print(mean_test_time)
