# -*- coding: utf-8 -*-

"""
This script starts or resumes the training of an autoencoder or an encoder that is 
defined in model_definitions.
It also contatins the adress of the training files and the epoch
"""


from keras.models import load_model
from keras import optimizers
from keras import backend as K
import numpy as np
import os
import sys
import argparse

from util.run_cnn import *
from model_definitions import *
from get_dataset_info import get_dataset_info


# start.py "vgg_1_xzt" 1 0 0 0 2 "up_down" True 0 11 18 50 1 
def parse_input():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('modeltag', type=str, help='e.g vgg_3-sgd; -XXX indicates version number and is ommited when looking up model by modeltag')
    parser.add_argument('runs', type=int)
    parser.add_argument("autoencoder_stage", type=int)
    parser.add_argument("autoencoder_epoch", type=int)
    parser.add_argument("encoder_epoch", type=int)
    parser.add_argument("class_type_bins", type=int)
    parser.add_argument("class_type_name", type=str)
    parser.add_argument("zero_center", type=int)
    parser.add_argument("verbose", type=int)    
    parser.add_argument("dataset", type=str, help="Name of test/training dataset to be used, eg xzt. n_bins is automatically selected")
    parser.add_argument("learning_rate", type=float)
    parser.add_argument("learning_rate_decay", default=0.05, type=float)
    parser.add_argument("epsilon", default=-1, type=int, help="Exponent of the epsilon used for adam.") #exponent of epsilon, adam default: 1e-08
    parser.add_argument("lambda_comp", type=int)
    parser.add_argument("optimizer", type=str)
    parser.add_argument("options", type=str)
    parser.add_argument("encoder_version", default="", nargs="?", type=str, help="e.g. -LeLu; Str added to the supervised file names to allow multiple runs on the same model.")
    
    args = parser.parse_args()
    params = vars(args)

    return params

params = parse_input()
modeltag = params["modeltag"]
runs=params["runs"]
autoencoder_stage=params["autoencoder_stage"]
autoencoder_epoch=params["autoencoder_epoch"]
encoder_epoch=params["encoder_epoch"]
class_type = (params["class_type_bins"], params["class_type_name"])
zero_center = params["zero_center"]
verbose=params["verbose"]
dataset = params["dataset"]
learning_rate = params["learning_rate"]
learning_rate_decay = params["learning_rate_decay"]
epsilon = params["epsilon"]
lambda_comp = params["lambda_comp"]
use_opti = params["optimizer"]
options = params["options"]
encoder_version=params["encoder_version"]
print(params)

"""
#Tag for the model used; Identifies both autoencoder and encoder
modeltag="vgg_1_xzt"

#How many additinal epochs the network will be trained for by executing this script:
runs=1

#Type of training/network
# 0: autoencoder
# 1: encoder+ from autoencoder w/ frozen layers
# 2: encoder+ from scratch, completely unfrozen
autoencoder_stage=2

#Define starting epoch of autoencoder model
autoencoder_epoch=0

#If in encoder stage (1 or 2), encoder_epoch is used to identify a possibly
#existing supervised-trained encoder network
encoder_epoch=0
#Define what the supervised encoder network is trained for, and how many neurons are in the output
#This also defines the name of the saved model
class_type = (2, 'up_down')

#Wheter to use a precalculated zero-center image or not
zero_center = True

#Verbose bar during training?
#0: silent, 1:verbose, 2: one log line per epoch
verbose=0

# x  y  z  t
# 11 13 18 50
n_bins = (11,18,50,1)

#Learning rate, usually 0.001
learning_rate = 0.001
"""

#Naming scheme for models
"""
Autoencoder
"trained_" + modeltag + "_supervised_" + class_type[1] + "_epoch" + encoder_epoch + ".h5" 

Encoder+
"trained_" + modeltag + "_autoencoder_" + class_type[1] + "_epoch" + encoder_epoch + ".h5"

Encoder+ new
"trained_" + modeltag + "_autoencoder_" + autoencoder_epoch + "_supervised_" + class_type[1] + "_epoch" + encoder_epoch + ".h5" 
"""


def execute_training(modeltag, runs, autoencoder_stage, epoch, encoder_epoch, class_type, zero_center, verbose, dataset, learning_rate, learning_rate_decay, epsilon, lambda_comp, use_opti, encoder_version, options):
    #Get info like path of trainfile etc.
    dataset_info_dict = get_dataset_info(dataset)
    home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"]
    
    
    #All models are now saved in their own folder   models/"modeltag"/
    model_folder = home_path + "models/" + modeltag + "/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    #Optimizer used in all the networks:
    lr = learning_rate # 0.01 default for SGD, 0.001 for Adam
    lr_decay = learning_rate_decay # % decay for each epoch, e.g. if 0.05 -> lr_new = lr*(1-0.05)=0.95*lr
    
    
    #automatically look for latest epoch if -1 was given:
    if autoencoder_stage==0 and epoch==-1:
        epoch=0
        while True:
            if os.path.isfile(model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(epoch+1) + '.h5')==True:
                epoch+=1
            else:
                break
    elif autoencoder_stage==1 and encoder_epoch == -1:
        encoder_epoch=0
        while True:
            if os.path.isfile(model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(epoch) +  "_supervised_" + class_type[1] + encoder_version + '_epoch' + str(encoder_epoch+1) + '.h5')==True:
                encoder_epoch+=1
            else:
                break
    elif autoencoder_stage==2 and encoder_epoch == -1:
        encoder_epoch=0
        while True:
            if os.path.isfile(model_folder + "trained_" + modeltag + "_supervised_" + class_type[1] + encoder_version + '_epoch' + str(encoder_epoch+1) + '.h5')==True:
                encoder_epoch+=1
            else:
                break
    elif autoencoder_stage==3 and encoder_epoch == -1:
        encoder_epoch=0
        while True:
            if os.path.isfile(model_folder + "trained_" + modeltag + "_autoencoder_supervised_parallel_" + class_type[1] + encoder_version + '_epoch' + str(encoder_epoch+1) + '.h5')==True:
                encoder_epoch+=1
            else:
                break
            
    #if lr is negative, take its absolute as the starting lr and apply the decays that happend during the
    #previous epochs; The lr gets decayed once when train_and_test_model is called (so epoch-1 here)
    if lr<0:
        if autoencoder_stage==0  and epoch>0:
            lr=abs(lr*(1-float(lr_decay))**(epoch-1))
            
        elif (autoencoder_stage==1 or autoencoder_stage==2 or autoencoder_stage==3)  and encoder_epoch>0:
            lr=abs(lr*(1-float(lr_decay))**(encoder_epoch-1))
        else:
            lr=abs(lr)
    
    #Optimizer to be used. If an epsilon is specified, adam is used with epsilon=10**(given epsilon).
    #only used when compiling model, so use lambda_comp if optimizer should be changed
    #Default:
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if use_opti == "adam" or use_opti=="ADAM":
        adam = optimizers.Adam(lr=lr,    beta_1=0.9, beta_2=0.999, epsilon=10**epsilon,   decay=0.0)
    elif use_opti == "SGD" or use_opti=="sgd":
        adam = optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    else:
        print("Optimizer ", use_opti, " unknown!")
        raise NameError(use_opti)
    
    #fit_model and evaluate_model take lists of tuples, so that you can give many single files (here just one)
    train_tuple=[[train_file, h5_get_number_of_rows(train_file)]]
    test_tuple=[[test_file, h5_get_number_of_rows(test_file)]]
    
    
    #Check wheter a file with this name exists or not
    def check_for_file(proposed_filename):
        if(os.path.isfile(proposed_filename)):
            sys.exit(proposed_filename+ "exists already!")
    
    #Zero-Center with precalculated mean image
    #xs_mean = np.load(zero_center_file) if zero_center is True else None
    n_gpu=(1, 'avolkov')
    if zero_center == True:
        xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=32, n_bins=n_bins, n_gpu=n_gpu[0])
    else:
        xs_mean = None
        print("Not using zero centering. Are you sure?")
    
    
    autoencoder_model=None
    #Setup network:
    #Autoencoder self-supervised training. Epoch is the autoencoder epoch, enc_epoch not relevant for this stage
    if autoencoder_stage==0:
        is_autoencoder=True
        modelname = modeltag + "_autoencoder"
        print("Autoencoder stage 0")
        if epoch == 0:
            #Create a new autoencoder network
            print("Creating new autoencoder network:", modeltag)
            model = setup_model(model_tag=modeltag, autoencoder_stage=0, modelpath_and_name=None, additional_options=options)
            model.compile(optimizer=adam, loss='mse')
            #Create header for new test log file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_log_file:
                metrics = model.metrics_names #["loss"]
                test_log_file.write('{0}\tTest {1}\tTrain {2}\tTime\tLR'.format("Epoch", metrics[0], metrics[0]))
            
        else:
            #Load an existing trained autoencoder network and train that
            autoencoder_model_to_load=model_folder + "trained_" + modelname + '_epoch' + str(epoch) + '.h5'
            print("Loading existing autoencoder to continue training:", autoencoder_model_to_load)
            if lambda_comp==False:
                model = load_model(autoencoder_model_to_load)
            elif lambda_comp==True:
                #in case of lambda layers: Load model structure and insert weights, because load model is bugged for lambda layers
                print("Lambda mode enabled")
                model=setup_model(model_tag=modeltag, autoencoder_stage=0, modelpath_and_name=None, additional_options=options)
                model.load_weights(autoencoder_model_to_load, by_name=True)
                model.compile(optimizer=adam, loss='mse')
                
                opti_weights=load_model(autoencoder_model_to_load).optimizer.get_weights()
                model.optimizer.set_weights(opti_weights)
            
    #Encoder supervised training:
    #Load the encoder part of an autoencoder, import weights from trained model, freeze it and add dense layers
    elif autoencoder_stage==1:
        print("Autoencoder stage 1")
        is_autoencoder=False
        #name of the autoencoder model file that the encoder part is taken from:
        autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(epoch) + '.h5'
        #name of the supervised model:
        modelname = modeltag + "_autoencoder_epoch" + str(epoch) +  "_supervised_" + class_type[1] + encoder_version
        
        if encoder_epoch == 0:
            #Create a new encoder network:
            print("Creating new encoder network", modeltag, "from autoencoder", autoencoder_model)
            model = setup_model(model_tag=modeltag, autoencoder_stage=1, modelpath_and_name=autoencoder_model, additional_options=options)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            #Create header for new test log file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_log_file:
                metrics = model.metrics_names #['loss', 'acc']
                test_log_file.write('{0}\tTest {1}\tTrain {2}\tTest {3}\tTrain {4}\tTime\tLR'.format("Epoch", metrics[0], metrics[0],metrics[1],metrics[1]))
            
        else:
            #Load an existing trained encoder network and train that
            encoder_network_to_load=model_folder + "trained_" + modelname + '_epoch' + str(encoder_epoch) + '.h5'
            print("Loading existing encoder network", encoder_network_to_load)
            model = load_model(encoder_network_to_load)
    
    
    #Unfrozen Encoder supervised training with completely unfrozen model:
    elif autoencoder_stage==2:
        print("Autoencoder stage 2")
        is_autoencoder=False
        #name of the supervised model:
        modelname = modeltag + "_supervised_" + class_type[1] + encoder_version
        
        if encoder_epoch == 0:
            #Create a new encoder network:
            print("Creating new unfrozen encoder network:", modelname)
            model = setup_model(model_tag=modeltag, autoencoder_stage=2, additional_options=options)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            #Create header for new test log file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_log_file:
                metrics = model.metrics_names #['loss', 'acc']
                test_log_file.write('{0}\tTest {1}\tTrain {2}\tTest {3}\tTrain {4}\tTime\tLR'.format("Epoch", metrics[0], metrics[0],metrics[1],metrics[1]))
            
        else:
            #Load an existing trained encoder network and train that
            load_this=model_folder + "trained_" + modelname + '_epoch' + str(encoder_epoch) + '.h5'
            print("Loading existing unfrozen encoder network", load_this)
            model = load_model(load_this)
    
    #Training of the supervised network on several different autoencoder epochs
    #epoch is calculated automatically and not used from user input
    #encoder epoch as usual
    #This does not use the same call for executing the training as stage 0,1 and 2
    
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    elif autoencoder_stage==3:
        #how many epochs should be trained on each autoencoder epoch, starting from epoch 1
        #if first epoch is 0, then the trained supervised network will be used
        how_many_epochs_each_to_train =[0,]+[2,]*10+[1,]*100
        #model to initialize from if first epoch is 0
        
        #this one is only used for vgg_3_eps modeltag
        init_model_eps=model_folder + "trained_vgg_3_eps_autoencoder_epoch1_supervised_up_down_accdeg2_epoch26.h5"
        
        print("Autoencoder stage 3:\nParallel training with epoch schedule:", how_many_epochs_each_to_train[:20], ",...")
        
        def switch_encoder_weights(encoder_model, autoencoder_model):
            #Change the weights of the frozen layers (up to the flatten layer) 
            #of the frozen encoder to that of another autoencoder model
            changed_layers=0
            for i,layer in enumerate(encoder_model.layers):
                if "flatten" not in layer.name:
                    layer.set_weights(autoencoder_model.layers[i].get_weights())
                    changed_layers+=1
                else:
                    break
            print("Weights of layers changed:", changed_layers)
        
        #Encoder epochs after which to switch the autoencoder model
        switch_autoencoder_model=np.cumsum(how_many_epochs_each_to_train)
        #calculate the current autoencoder epoch automatically based on the encoder epoch
        for ae_epoch,switch in enumerate(switch_autoencoder_model):
            if encoder_epoch-switch <= 0:
                autoencoder_epoch=ae_epoch+1
                break
        
        is_autoencoder=False
        #name of the autoencoder model file that the encoder part is taken from:
        autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(autoencoder_epoch) + '.h5'
        #name of the supervised model:
        modelname = modeltag + "_autoencoder_supervised_parallel_" + class_type[1] + encoder_version
        
        if encoder_epoch == 0:
            #Create a new encoder network:
            model = setup_model(model_tag=modeltag, autoencoder_stage=1, modelpath_and_name=autoencoder_model, additional_options=options )
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            
            #Custom model is loaded as initialization
            if switch_autoencoder_model[0]==0:
                if modeltag=="vgg_3_eps":
                    init_model=init_model_eps
                else:
                    raise("Cannot load initial model "+init_model+" Modeltags are different "+modeltag)
                print("Initializing model to", init_model)
                autoencoder_epoch=2
                autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(autoencoder_epoch) + '.h5'
                model_for_init = load_model(init_model)
                for i,layer in enumerate(model.layers):
                    layer.set_weights(model_for_init.layers[i].get_weights())
            
            
            #Create header for new test log file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_log_file:
                metrics = model.metrics_names #['loss', 'acc']
                test_log_file.write('{0}\tTest {1}\tTrain {2}\tTest {3}\tTrain {4}\tTime\tLR'.format("Epoch", metrics[0], metrics[0],metrics[1],metrics[1]))
            
        else:
            #Load an existing trained encoder network and train that
            model = load_model(model_folder + "trained_" + modelname + '_epoch' + str(encoder_epoch) + '.h5')
        
                
        #Own execution of training
        #Set LR of loaded model to new lr
        K.set_value(model.optimizer.lr, lr)
            
        #Which epochs are the ones relevant for current stage
        running_epoch=encoder_epoch
            
        model.summary()
        print("Model: ", modelname)
        print("Current State of optimizer: \n", model.optimizer.get_config())
        print("Train files:", train_tuple)
        print("Test files:", test_tuple)
        print("Using autoencoder model:", autoencoder_model)
        
        #Execute Training:
        for current_epoch in range(running_epoch,running_epoch+runs):
            #Does the model we are about to save exist already?
            check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
            
            #Train network, write logfile, save network, evaluate network, save evaluation to file
            lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                                 batchsize=32, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, epoch=current_epoch,
                                 shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                                 save_path=model_folder, is_autoencoder=is_autoencoder, verbose=verbose, broken_simulations_mode=broken_simulations_mode)  
            
            if current_epoch+1 in switch_autoencoder_model:
                autoencoder_epoch+=1
                autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(autoencoder_epoch) + '.h5'
                print("Changing weights after epoch ",current_epoch+1," to ",autoencoder_model)
                switch_encoder_weights(model, load_model(autoencoder_model))
                
        sys.exit()
        
        
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        

        
    #Set LR of loaded model to new lr
    K.set_value(model.optimizer.lr, lr)
    
        
    #Which epochs are the ones relevant for current stage
    if is_autoencoder==True:
        running_epoch=epoch #Stage 0
    else:
        running_epoch=encoder_epoch #Stage 1 and 2
        
    model.summary()
    print("Model: ", modelname)
    print("Current State of optimizer: \n", model.optimizer.get_config())
    print("Train files:", train_tuple)
    print("Test files:", test_tuple)
    if autoencoder_model is not None: print("Using autoencoder model:", autoencoder_model)
    
    #Execute Training:
    for current_epoch in range(running_epoch,running_epoch+runs):
        #Does the model we are about to save exist already?
        check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
        
        #print all the input for train_and_test_model for debugging
        """
        print("model=",model, "modelname=",modelname, "train_files=",train_tuple, "test_files=",test_tuple,
                             "batchsize=32, n_bins=",n_bins, "class_type=",class_type, "xs_mean=",xs_mean, "epoch=",current_epoch,
                             "shuffle=False, lr=",lr, "lr_decay=",lr_decay, "tb_logger=False, swap_4d_channels=None,",
                             "save_path=",model_folder, "is_autoencoder=",is_autoencoder, "verbose=",verbose)
        """
        
        #Train network, write logfile, save network, evaluate network, save evaluation to file
        lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                             batchsize=32, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, epoch=current_epoch,
                             shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                             save_path=model_folder, is_autoencoder=is_autoencoder, verbose=verbose, broken_simulations_mode=broken_simulations_mode)    
    
    
    

execute_training(modeltag, runs, autoencoder_stage, autoencoder_epoch, encoder_epoch, class_type, zero_center, verbose, dataset, learning_rate, learning_rate_decay=learning_rate_decay, epsilon=epsilon, lambda_comp=lambda_comp, use_opti=use_opti, encoder_version=encoder_version, options=options)
