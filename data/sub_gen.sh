#!/bin/bash -l
#
# allocate 16 nodes (64 CPUs) for 6 hours
#
# job name
#PBS -N generate
#
# stdout and stderr files
#
# first non-empty non-comment line ends PBS options

cd $WOODYHOME/Km3-Autoencoder/data

python generate.py
