# Deep convolutional autoencoders for KM3NeT/ORCA
This repository contains scripts for building, training and visualizing autoencoder-based networks. These are the most important files:

## Training the network
The main file for training and testing networks is **scripts/run_autoencoder.py**. 
Models will be saved after every epoch, and log files will be created which show the history of training. 
The test log file _test.txt contains the summarized statistics of every epoch, while the train log files _log.txt contain the batch-wise statistics of every single epoch.
The script has plenty of input parameters, so the use of a shell script to run it is recommended.
A detailled explanation for all the parameters is given in the example shell script **submit.sh**, but the most important ones are the two following:

#### modeltag
Defines the architecture of the model that will be used. They are defined by the files in the folder scripts/model_def/.

#### autoencoder_stage
Describes the type of network used for training. The basic stages are:
- 0 = Autoencoder
- 1 = Encoder+dense: Take the encoder from an autoencoder, freeze it and add dense layers 
- 2 = Supervised approach: Like the encoder+dense, but not frozen and randomly initialized. This is essentially a standard supervised network.
- 3 = Successive training: Like stage 1, but switch out the weights of the encoder regularily. This way, one can scan which encoder is best for the supervised task.


## Visualization
These are scripts used for visualizing the progress of the training.

- plot_statistics_parser: Plot the train and test loss of a model over the training epoch, based on test and train log files. Can also plot multiple models, given they have the same loss.
- plots_statistics_parallel: Plot the successive training of a model. This will show both the autoencoder as well as the encoder+dense (stage 3) train and test losses, both plotted over the autoencoder epoch. ![Shows the performance of an autoencoder and the corresponding encoder+dense network during training.](results/plots/readme_examples/statistics_parallel_trained_vgg_5_morefilter_autoencoder_supervised_parallel_up_down_test.pdf?raw=true "Successive training history")
- scripts/make_quick_autoencoder_reco_plots: Show the reconstruction of an autoencoder. This will plot an event and the output from a saved autoencoder which is used on this event side by side as a histogram. ![Autoencoder reconstruction at an early point during training.](results/plots/readme_examples/AE_vgg_3_epoch_10_reko.pdf?raw=true "Reconsturction")

