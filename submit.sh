#!/bin/bash -l
#
# allocate 16 nodes (64 CPUs) for 6 hours
#PBS -l walltime=00:59:00, nodes=1:ppn=4:gtx1080
#
# job name 
#PBS -N autoencoder_test
#
# stdout and stderr files
#PBS -o job33.out -e job33.err
#
# first non-empty non-comment line ends PBS options

# jobs always start in $HOME -
# change to a temporary job directory on $FASTTMP
mkdir ${FASTTMP}/$PBS_JOBID
cd ${FASTTMP}/$PBS_JOBID
# copy input file from location where job was submitted
cp ${PBS_O_WORKDIR}/inputfile .


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



# run
mpirun ${WOODYHOME}/bin/a.out -i inputfile -o outputfile

# save output on parallel file system
mkdir -p ${FASTTMP}/output/$PBS_JOBID
cp outputfile ${FASTTMP}/output/$PBS_JOBID
cd 
# get rid of the temporary job dir
rm -rf ${FASTTMP}/$PBS_JOBID