# -*- coding: utf-8 -*-

"""
This script starts or resumes the training of an autoencoder or an encoder that is 
defined in model_definitions. Due to the large amout of input arguments,
this should be run from a shell script.
A documentation for the possible inputs is given in submit.sh.
"""


from keras.models import load_model
from keras import backend as K
import numpy as np
import os
import argparse

from util.run_cnn import (
        train_and_test_model, load_zero_center_data, h5_get_number_of_rows, 
        setup_learning_rate, get_autoencoder_loss, look_up_latest_epoch, 
        get_supervised_loss_and_metric, setup_autoencoder_model, 
        setup_encoder_dense_model, setup_optimizer, setup_successive_training, 
        switch_encoder_weights, make_encoder_stateful, unfreeze_conv_layers,
        lr_schedule
        )
from get_dataset_info import get_dataset_info


def unpack_parsed_args():
    """ 
    Read out input arguments and return them as a tuple for the training
    
    They will be printed and returned as a dict, so that they can be handed to
    execute_training.
    
    Returns:
        params: Dict with all input parameters of the parser.
    """
    parser = argparse.ArgumentParser(description='The main function for training \
        autoencoder-based networks. See submit.sh for detailed explanations of all parameters.')
    parser.add_argument('modeltag', type=str, 
                        help='e.g vgg_3-sgd; -XXX indicates version number and \
                        is ommited when looking up model by modeltag')
    parser.add_argument('runs', 
                        help="How many new epochs should be trained by executing this script..", type=int)
    parser.add_argument("autoencoder_stage", 
                        help="Stage of autoencoder training: 0 for AE, 1 for enc,\
                        2 for unfrozen, 3 for parallel", type=int)
    parser.add_argument("autoencoder_epoch",
                        help="Epoch of AE network to be used", type=int)
    parser.add_argument("encoder_epoch", 
                        help="Epoch of encoder network to be used", type=int)
    parser.add_argument("class_type_bins", type=int)
    parser.add_argument("class_type_name", help="Name of target", type=str)
    parser.add_argument("zero_center", type=int)
    parser.add_argument("verbose", type=int)    
    parser.add_argument("dataset", type=str, 
                        help="Name of test/training dataset to be used, \
                        eg xzt. n_bins is automatically selected")
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("learning_rate_decay", help="LR decay per epoch \
                        (multipl.), or name of lr schedule")
    parser.add_argument("epsilon", type=int, help="Exponent of the \
                        epsilon used for adam.") #adam default: 1e-08
    parser.add_argument("lambda_comp", type=int)
    parser.add_argument("optimizer", type=str)
    parser.add_argument("options", type=str)
    parser.add_argument("encoder_version", default="", nargs="?", type=str,
                        help="e.g. -LeLu; Str added to the supervised file names\
                        to allow multiple runs on the same model.")
    parser.add_argument("--ae_loss_name", default="mse", nargs="?", type=str, 
                        help="Loss that is used during AE training. Default is mse.")
    parser.add_argument("--supervised_loss", default="auto", nargs="?", type=str, 
                        help="Loss that is used during supervised training.\
                        Default is 'auto', which is based on the number of output neurons.")
    parser.add_argument("--init_model", default=None, nargs="?", 
                        help="Path to a model that is used for initializing.")
    
    args = parser.parse_args()
    params = vars(args)
    
    print("\nArguments handed to parser:")
    for keyword in params:
        print(keyword, ":\t", params[keyword])
    print("\n")
    
    return params
   


def extra_autoencoder_stages_setup(autoencoder_stage, ae_loss_name, supervised_loss):
    """
    Setup for special trainings that dont fit into regular categories.
    
    They still use the same network architectures as the standard categories,
    but have some additional properties during training.
    
    Returns:
        autoencoder_stage: Defines which architecture is loaded.
        ae_loss_name: Loss of autoencoder training.
        supervised_loss. Loss of supervised training.
        unfreeze_layer_training: Bool: Parts of the network will be gradually unfrozen 
            during training.
        is_AE_adevers_training: int defining the stage of adversarial autoencoder training.
    """
    
    if autoencoder_stage==4:
        print("Autoencoder stage 4: Unfreeze Training. Setting up network like in AE stage 1...")
        autoencoder_stage=1
        unfreeze_layer_training = True
    else:
        unfreeze_layer_training = False
    #If autoencoder stage 5,6 or 7 is selected: Adversarial autoencoders
    #AAE training: similar to stage 0, but with other loss and training setup
    #is_AE_adevers_training=0 no AAE training
    #1 or 2: train critic and generator alternating
    #3: Train only critic
    if autoencoder_stage==5:
        print("Starting AE training in adversarial setup (stage 5). Loss will\
              be cat cross entropy and labels will eb fixed! Otherwise like stage 0.")
        #for adversarial AE training, setup like normal autoencoder
        autoencoder_stage=0
        ae_loss_name = "categorical_crossentropy"
        supervised_loss = "cat_cross_inv"
        is_AE_adevers_training=1
    elif autoencoder_stage==6:
        #preperation for AAE training: train only the critic
        autoencoder_stage=0
        ae_loss_name = "categorical_crossentropy"
        supervised_loss = None
        is_AE_adevers_training=3
    elif autoencoder_stage==7:
        #preperation for AAE training: train only the critic
        autoencoder_stage=0
        ae_loss_name = "categorical_crossentropy"
        supervised_loss = None
        is_AE_adevers_training=4
    else:
        is_AE_adevers_training=False
        unfreeze_layer_training=False
        
    return autoencoder_stage, ae_loss_name, supervised_loss, unfreeze_layer_training, is_AE_adevers_training




def build_model(autoencoder_stage, modeltag, epoch, optimizer, ae_loss, 
                options, custom_objects, model_folder, class_type,
                encoder_version, encoder_epoch, supervised_loss,
                supervised_metrics, init_model_path):
    """
    Construct and compile the model. Also returns info for successive training.    
    
    The basic architectures are:
        Stage 0: Autoencoder
        Stage 1: Encoder+dense w/ frozen encoder, initialized to give AE
    Variants of the above are:
        Stage 2: Encoder+dense completely unfrozen, random initialized.
        Stage 3: Like stage 1, but trained successively.
            
    Returns:
        model : The keras model to train.
        modelname : Name of that model as a string.
        autoencoder_model : For encoder+dense networks:
            Path of the autoencoder model file that the encoder part is taken from.
        is_autoencoder : Whether the model is an autoencoder or not.
        last_encoder_layer_index_override : Manual definition of the layer 
            that is the bottleneck.
        switch_autoencoder_model : Successive training: When to switch the encoder weights.
        succ_autoencoder_epoch : Successive training: The weights of which autoencoder 
            are being used right now.
    """
    #Setup network:
    #If epoch is 0, a new model is created. Otherwise, the 
    #existing model of the given epoch is loaded unchanged.
    
    number_of_output_neurons = int(class_type[0])
    #Only relevant for stage 3: successive training:
    (last_encoder_layer_index_override, switch_autoencoder_model, 
     succ_autoencoder_epoch) = None, None, None
     
    #Autoencoder self-supervised training. Epoch is the autoencoder epoch, 
    #enc_epoch not relevant for this stage
    if autoencoder_stage==0:
        is_autoencoder=True
        modelname = modeltag + "_autoencoder"
        autoencoder_model = modelname
        print("\n\nAutoencoder stage 0")
        model = setup_autoencoder_model(modeltag, epoch, optimizer, ae_loss, 
                                        options, custom_objects, model_folder, modelname)
    #Encoder+dense supervised training:
    #Load the encoder part of an autoencoder, import weights from 
    #trained model, freeze it and add dense layers
    elif autoencoder_stage==1:
        print("\n\nAutoencoder stage 1")
        is_autoencoder=False
        #name of the autoencoder model file that the encoder part is taken from:
        autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(epoch) + '.h5'
        #name of the supervised model:
        modelname = modeltag + "_autoencoder_epoch" + str(epoch) +  "_supervised_" + class_type[1] + encoder_version
        
        model = setup_encoder_dense_model(modeltag, encoder_epoch, modelname, autoencoder_stage,
                              number_of_output_neurons, supervised_loss, supervised_metrics,
                              optimizer, options, model_folder, custom_objects,
                              autoencoder_model)
    #Unfrozen Encoder supervised training with completely unfrozen model.
    #No weights of autoencoders will be loaded in.
    elif autoencoder_stage==2:
        print("\n\nAutoencoder stage 2")
        is_autoencoder=False
        autoencoder_model=None
        #name of the supervised model:
        modelname = modeltag + "_supervised_" + class_type[1] + encoder_version
        
        model = setup_encoder_dense_model(modeltag, encoder_epoch, modelname, autoencoder_stage,
                              number_of_output_neurons, supervised_loss, supervised_metrics,
                              optimizer, options, model_folder, custom_objects,
                              autoencoder_model)
    #Training of the supervised network on several different autoencoder epochs
    #epoch is calculated automatically and not used from user input
    #encoder epoch as usual
    elif autoencoder_stage==3:
        print("\n\nAutoencoder stage 3")
        is_autoencoder=False
        (switch_autoencoder_model, succ_autoencoder_epoch, make_stateful, 
         last_encoder_layer_index_override) = setup_successive_training(
                              modeltag, encoder_epoch)
        #name of the autoencoder model file that the encoder part is taken from:
        autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(succ_autoencoder_epoch) + '.h5'
        #name of the supervised model:
        modelname = modeltag + "_autoencoder_supervised_parallel_" + class_type[1] + encoder_version
    
        #Setup encoder+dense and load encoder weights (like autoencoder_stage=1)
        model = setup_encoder_dense_model(modeltag, encoder_epoch, modelname, 1,
                          number_of_output_neurons, supervised_loss, supervised_metrics,
                          optimizer, options, model_folder, custom_objects,
                          autoencoder_model)
        
        if make_stateful==True:
            model = make_encoder_stateful(model)
              
    #Initialize the model with the weights of a saved one
    if init_model_path is not None and init_model_path != "None":
        print("Initializing model weights to", init_model_path)
        init_model = load_model(init_model_path, custom_objects=custom_objects)
        for i,layer in enumerate(model.layers):
                layer.set_weights(init_model.layers[i].get_weights())
            
    return (model, modelname, autoencoder_model, is_autoencoder,
            last_encoder_layer_index_override, switch_autoencoder_model, 
            succ_autoencoder_epoch, )




def train_model(model, dataset, zero_center, modelname, autoencoder_model, 
                lr_schedule_number, runs, learning_rate, model_folder, 
                last_encoder_layer_index_override, switch_autoencoder_model,
                autoencoder_stage, succ_autoencoder_epoch, modeltag,
                unfreeze_layer_training, custom_objects, class_type, lr_decay, 
                verbose, is_AE_adevers_training, is_autoencoder, epoch,
                encoder_epoch):
    """ Train, test and save the model and logfiles. """
    #Get infos about the dataset
    dataset_info_dict = get_dataset_info(dataset)
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
    filesize_factor=dataset_info_dict["filesize_factor"]
    filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
    
    #The files to train and test on, together with the nummber of events in them
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    
    #Zero-Center for the data. if zero center image does not exist, a new one
    #is calculated and saved
    if zero_center == True:
        xs_mean = load_zero_center_data(train_files=train_tuple, 
                                        batchsize=batchsize, n_bins=n_bins, n_gpu=1)
        print("Using zero centering.")
    else:
        xs_mean = None
        print("Not using zero centering.")
        
    #Which epochs are the ones relevant for current stage
    if is_autoencoder==True:
        running_epoch=epoch #Stage 0
    else:
        running_epoch=encoder_epoch #Stage 1,2,3
        
    #Set LR of loaded model to new lr
    if lr_schedule_number != None:
            lr=lr_schedule(running_epoch+1, lr_schedule_number, learning_rate )
    K.set_value(model.optimizer.lr, lr)
    
    #Print info about the model and training
    model.summary()
    print("\n\nModel: ", modelname)
    print("Current State of optimizer: \n", model.optimizer.get_config())
    filesize_hint="Filesize factor="+str(filesize_factor) if filesize_factor!=1 else ""
    filesize_hint_test="Filesize factor test="+str(filesize_factor_test) if filesize_factor_test!=1 else ""
    print("Train files:", train_tuple, filesize_hint)
    print("Test files:", test_tuple, filesize_hint_test)
    print("Using metrics:", model.metrics_names)
    if autoencoder_model is not None: print("Using autoencoder model:", autoencoder_model)
    
    
    #Main loop: Execute Training
    for current_epoch in range(running_epoch,running_epoch+runs):
        print("\n")
        #This is before epoch current_epoch+1
        #Does the model we are about to save exist already?
        proposed_filename = model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5'
        if(os.path.isfile(proposed_filename)):
            raise NameError("Warning:", proposed_filename+ "exists already!")
                
        if lr_schedule_number != None:
            lr=lr_schedule(current_epoch+1, lr_schedule_number, learning_rate )
            K.set_value(model.optimizer.lr, lr)
            
        #Autoencoder stage 3: Successive training
        #Load in weights of new encoders periodically
        #succ_autoencoder_epoch is the epoch of the autoencoder from which
        #the weights are loaded in
        if autoencoder_stage==3:
            if current_epoch in switch_autoencoder_model:
                succ_autoencoder_epoch+=1
                autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(succ_autoencoder_epoch) + '.h5'
                print("Changing weights before epoch ",current_epoch+1," to ",autoencoder_model)
                switch_encoder_weights(model, load_model(autoencoder_model, 
                                                custom_objects=custom_objects), 
                                        last_encoder_layer_index_override)
            
        #Autoencoder stage 4: Layer unfreeze training
        if unfreeze_layer_training==True:
            #Unfreeze C layers of the model according to schedule
            #An additional C block is set trainable before these epochs
            unfreeze_a_c_block_at = np.array([5,10,15,20,25,30,35,40,])
            
            how_many = np.where(unfreeze_a_c_block_at==current_epoch)[0]
            if len(how_many)>0:
                how_many=how_many[0]+1
                model = unfreeze_conv_layers(model, how_many)
            
            
        #Train network, write logfile, save network, evaluate network, save evaluation to file
        lr = train_and_test_model(model=model, 
                                  modelname=modelname, 
                                  train_files=train_tuple, 
                                  test_files=test_tuple,
                                  batchsize=batchsize,
                                  n_bins=n_bins, 
                                  class_type=class_type, 
                                  xs_mean=xs_mean, 
                                  epoch=current_epoch,
                                  shuffle=False, 
                                  lr=lr, 
                                  lr_decay=lr_decay, 
                                  tb_logger=False,
                                  swap_4d_channels=None,
                                  save_path=model_folder,
                                  is_autoencoder=is_autoencoder,
                                  verbose=verbose,
                                  broken_simulations_mode=broken_simulations_mode,
                                  dataset_info_dict=dataset_info_dict, 
                                  is_AE_adevers_training=is_AE_adevers_training)    

    

def execute_network_training():
    """ Main function for training autoencoder-based networks. """
    params = unpack_parsed_args()
    modeltag = params["modeltag"]
    runs=params["runs"]
    autoencoder_stage=params["autoencoder_stage"]
    epoch=params["autoencoder_epoch"]
    encoder_epoch=params["encoder_epoch"]
    class_type = (params["class_type_bins"], params["class_type_name"])
    zero_center = params["zero_center"]
    verbose=params["verbose"]
    dataset = params["dataset"]
    learning_rate = params["learning_rate"]
    learning_rate_decay = params["learning_rate_decay"]
    epsilon = params["epsilon"]
    #lambda_comp = params["lambda_comp"] not supported anymore
    use_opti = params["optimizer"]
    options = params["options"]
    encoder_version = params["encoder_version"]
    ae_loss_name=params["ae_loss_name"]
    supervised_loss=params["supervised_loss"]
    init_model_path=params["init_model"]
    
    
    #Every model architecture is identified by its modeltag
    #Each will be given its own directory in the main_folder automatically
    #e.g. if the modeltag is "vgg_3", all saved models will 
    #be stored in: main_folder+"/vgg_3/"
    main_folder = "/home/woody/capn/mppi013h/Km3-Autoencoder/models"
    
    #Create the folder for the model if it does not exist already
    model_folder = main_folder + "/" + modeltag + "/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        print("Created model folder", model_folder)
    
    #For AE stages other than 0,1,2,3:
    (autoencoder_stage, ae_loss_name, supervised_loss, unfreeze_layer_training, 
     is_AE_adevers_training) = extra_autoencoder_stages_setup(autoencoder_stage, 
                           ae_loss_name, supervised_loss)
    
    #Number of output neurons of the supervised networks (AE ignore this)
    number_of_output_neurons = int(class_type[0])
    if number_of_output_neurons<1:
        raise ValueError("number_of_output_neurons have to be >= 1")
    
    #define the loss function to use for a new AE
    #(saved autoencoders will continue to use their original one)
    ae_loss, custom_objects = get_autoencoder_loss(ae_loss_name)
    print("Using autoencoder loss:", ae_loss_name)
    
    #Define the loss function and additional metrics to use for a new 
    #Encoder+dense network (saved nets will continue to use their original one)
    supervised_loss, supervised_metrics = get_supervised_loss_and_metric(supervised_loss, 
                                                        number_of_output_neurons)
    print("Using supervised loss:", supervised_loss)
    
    #Setup learning rate for the start of the training
    lr, lr_decay, lr_schedule_number = setup_learning_rate(learning_rate, 
                                        learning_rate_decay, autoencoder_stage, 
                                        epoch, encoder_epoch)
    if lr_schedule_number!= None:
        print("Using learning rate schedule", lr_schedule_number)
    
    #automatically look for the epoch of the most recent saved model 
    #of the current architecture if epoch=-1 was given:
    epoch, encoder_epoch = look_up_latest_epoch(autoencoder_stage, epoch, 
                                        encoder_epoch, model_folder, modeltag, 
                                        class_type, encoder_version)
    #Optimizer for training:
    optimizer = setup_optimizer(use_opti, lr, epsilon)
    
    #Construct, initialize and compile model:
    (
     model, modelname, autoencoder_model, is_autoencoder,
     last_encoder_layer_index_override, switch_autoencoder_model, 
     succ_autoencoder_epoch, 
     ) = build_model(
             autoencoder_stage, modeltag, epoch, optimizer, ae_loss, 
             options, custom_objects, model_folder, class_type,
             encoder_version, encoder_epoch, supervised_loss,
             supervised_metrics, init_model_path)
    
    #Train, test, save the model:
    train_model(model, dataset, zero_center, modelname, autoencoder_model, 
                lr_schedule_number, runs, learning_rate, model_folder, 
                last_encoder_layer_index_override, switch_autoencoder_model,
                autoencoder_stage, succ_autoencoder_epoch, modeltag,
                unfreeze_layer_training, custom_objects, class_type, lr_decay, 
                verbose, is_AE_adevers_training, is_autoencoder, epoch,
                encoder_epoch)
    
    
if __name__ == "__main__":
    execute_network_training()

    