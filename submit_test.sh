#!/bin/bash -l
#
# allocate 16 nodes (64 CPUs) for 6 hours
#PBS -l walltime=00:59:00, nodes=1:ppn=4
#
# job name 
#PBS -N autoencoder_test
#
# stdout and stderr files
#PBS -o job33.out -e job33.err
#
# first non-empty non-comment line ends PBS options


module load python/3.5-anaconda
#content of env_cnn.sh
# load conda virtualenv
source activate /home/hpc/capn/mppi013h/python_conda_environment/neuralnets
# required:
module use -a /home/vault/capn/shared/apps/U16/modules
module load cudnn/6.0-cuda8.0
# obtaining software from the outside world requires a working proxy to be set
export https_proxy=https://pi4060.physik.uni-erlangen.de:8888
export http_proxy=http://pi4060.physik.uni-erlangen.de:8888
#pip install --user --upgrade tensorflow-gpu keras h5py numpy sklearn
pip install --upgrade tensorflow-gpu keras h5py numpy sklearn

python first_autoencoder.py