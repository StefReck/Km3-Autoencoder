# Deep convolutional autoencoders for KM3NeT/ORCA
This repository contains scripts for building, training and visualizing autoencoder-based networks. 
They are used on simulated data of the KM3NeT/ORCA neutrino telescope.
See also [OrcaNet](https://github.com/ViaFerrata/OrcaNet) for a project with supervised deep networks for ORCA.

The basic idea is to train an autoencoder unsupervised first, which could in principle be done on measured data. 
This way, one can circumvent possibly existing deviations between simulations and measured data.

Then, the encoder part of the autoencoder is taken, its weights are frozen and dense layers are added. 
This encoder+dense net is trained on simulations to predict the desired feature of the data.
![Autoencoder training procedure](results/plots/readme_examples/autoencoder_principle.png?raw=true "Autoencoder training procedure")

## Training the network
The main file for training and testing networks is **scripts/run_autoencoder.py**. 
Models will be saved after every epoch, and log files will be created which show the history of training. 
The test log file xxx_test.txt contains the summarized statistics of the whole training, while the train log files xxx_log.txt contain the batch-wise statistics of every single epoch.
The script has plenty of input parameters, so the use of a shell script to run it is recommended.
A detailed explanation of all the parameters is given in the example shell script **submit.sh**, but the most important ones are the following:

#### modeltag
Defines the architecture of the model that will be used. They are defined by the files in the folder **scripts/model_def/**.

#### autoencoder_stage
Describes the type of network used for training. The basic stages are:
- 0 = Autoencoder
- 1 = Encoder+dense: Take the encoder from an already trained and saved autoencoder, freeze it and add dense layers 
- 2 = Supervised approach: Has the same architecture as the encoder+dense network, but does not load in the weights from an existing autoencoder and is not frozen. This is essentially a standard supervised network.
- 3 = Successive training: Like autoencoder_stage 1, but load in the weights of new encoders every few epochs (more often in later epochs). This way, one can scan which encoder is best for the encoder+dense supervised task.

There are also some additional autoencoder_stages for advanced trainings (like GANs), but they are in an experimental state.

## Visualization
These are scripts used for visualizing the progress of the training.

- plot_statistics_parser.py: Plot the train and test loss of a model over the training epoch, based on test and train log files. Can also plot multiple models, given that they have the same type of loss.
- plots_statistics_parallel.py: Plot the successive training of a model. This will show both the autoencoder as well as the encoder+dense (autoencoder_stage=3) train and test losses, each plotted over the autoencoder epoch. ![Shows the performance of an autoencoder and the corresponding encoder+dense network during training.|50%](results/plots/readme_examples/statistics_parallel_trained_vgg_5_morefilter_autoencoder_supervised_parallel_up_down_test.png?raw=true "Successive training history")
- scripts/make_quick_autoencoder_reco_plots.py: Show the reconstruction of an autoencoder. This will plot an event and the output from a saved autoencoder which is applied on this event side by side as histograms. ![Autoencoder reconstruction at an early point during training.|50%](results/plots/readme_examples/AE_vgg_3_epoch_10_reko.png?raw=true "Reconstruction")

