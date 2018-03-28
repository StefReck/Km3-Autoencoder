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

from util.run_cnn import train_and_test_model, load_zero_center_data, h5_get_number_of_rows
from model_definitions import setup_model
from get_dataset_info import get_dataset_info
from util.custom_loss_functions import get_custom_objects


# start.py "vgg_1_xzt" 1 0 0 0 2 "up_down" True 0 11 18 50 1 
# read out input arguments and return them as a tuple for the training
def unpack_parsed_args():
    parser = argparse.ArgumentParser(description='The main function for training autoencoder-base networks.')
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
    parser.add_argument("learning_rate_decay")
    parser.add_argument("epsilon", type=int, help="Exponent of the epsilon used for adam.") #adam default: 1e-08
    parser.add_argument("lambda_comp", type=int)
    parser.add_argument("optimizer", type=str)
    parser.add_argument("options", type=str)
    parser.add_argument("encoder_version", default="", nargs="?", type=str, help="e.g. -LeLu; Str added to the supervised file names to allow multiple runs on the same model.")
    
    args = parser.parse_args()
    params = vars(args)
    print(params)
    
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
    
    return modeltag, runs, autoencoder_stage, autoencoder_epoch, encoder_epoch, class_type, zero_center, verbose, dataset, learning_rate, learning_rate_decay, epsilon, lambda_comp, use_opti, encoder_version, options
   
"""
# Tag for the model used; Identifies both autoencoder and encoder
# This also defines the name of the folder where everything is saved
modeltag="vgg_3"
#How many additinal epochs the network will be trained for by executing this script:
runs=10
#Type of training/network
# 0: autoencoder
# 1: encoder+ from autoencoder w/ frozen layers
# 2: encoder+ from scratch, completely unfrozen
# 3: Parallel supervised training (load in new frozen encoder epoch according to schedule)
autoencoder_stage=1
#Define starting epoch of autoencoder model (stage 1: which frozen encoder epoch to use)
# not used for stage 2 and 3
autoencoder_epoch=50
#If in encoder stage (1 or 2), encoder_epoch is used to identify a possibly
#existing supervised-trained encoder network; not used for stage 0
encoder_epoch=-1
#Define what the supervised encoder network is trained for, and how many neurons are in the output
#This also defines the name of the saved model
# class type names are currently "up_down", "muon-CC_to_elec-CC"
class_type_bins=2
class_type_name="up_down"
#Wheter to use a precalculated zero-center image or not (1,0)
#should typically be 1
zero_center=1
#Verbose bar during training?
#0: silent, 1:verbose, 2: one log line per epoch
verbose=2
# which dataset to use; see get_dataset_info.py
# x  y  z  t  c
# 11 13 18 50 31
#additional options can be appended via -XXX-YYY, e.g. "xzt-filesize=0.3"
dataset="xzt"
#Initial Learning rate, usually 0.001
# negative lr= take the absolute as the lr at epoch 1, and apply the decay epoch-times
learning_rate=-0.001
#lr_decay can either be a float, e.g. 0.05 for 5% decay of lr per epoch,
#or it can be a string like s1 for lr schedule 1. In this case, lr is ignored
learning_rate_decay=0.05

# exponent of epsilon for the adam optimizer (actualy epsilon is 10^this)
epsilon=-8
# lambda compatibility mode (0,1): Load the model manually from model definition,
# and insert the weigths; can be used to overwrite parameters of the optimizer
lambda_comp=0
#optimizer to use, either "adam" or "sgd"
optimizer="adam"
# allows to create multiple supervised trainings from the same AE model
# the version is added to the filename
encoder_version=""
# Additional options for the model, see model_definitions.py
# e.g. "dropout=0.3" or "unlock_BN"
options="None"

cd $WOODYHOME/Km3-Autoencoder
. ./../env_cnn.sh

python scripts/run_autoencoder.py $modeltag $runs $autoencoder_stage $autoencoder_epoch $encoder_epoch $class_type_bins $class_type_name $zero_center $verbose $dataset $learning_rate $learning_rate_decay $epsilon $lambda_comp $optimizer $options $encoder_version
"""


def lr_schedule(before_epoch, lr_schedule_number, learning_rate):
    #learning rate is the original lr input
    #return the desired lr of an epoch according to a lr schedule.
    #In the test_log file, the epoch "before_epoch" will have this lr.
    #lr rate should be set to this before starting the next epoch.
    if lr_schedule_number=="s1":
        #decay lr by 5 percent/epoch for 13 epochs down to half, then constant
        #for parallel training
        if before_epoch<=14:
            lr=0.001 * 0.95**(before_epoch-1)
        else:
            lr=0.0005
    
    elif lr_schedule_number=="s2":
        #for autoencoder training
        if before_epoch<=20:
            lr=0.001
        else:
            lr=0.01
        
    elif lr_schedule_number=="s3":
        #for autoencoder training
        if before_epoch<=20:
            lr=0.01
        else:
            lr=0.1
        
    elif lr_schedule_number=="steps15":
        # multiply user lr by 10 every 15 epochs
        start_lr       = learning_rate
        multiply_with  = 10
        every_n_epochs = 15
        lr = start_lr * multiply_with**np.floor((before_epoch-1)/every_n_epochs)
        
    elif lr_schedule_number=="c15red":
        # used for e.g. vgg5 200 large and vgg5 200 deep
        # constant 0.1, after epoch 15 increase by 10% per epoch for 5 epochs,
        # then constant again
        if before_epoch<=15:
            lr = 0.1
        elif before_epoch<=27:
            lr = 0.1 * 1.1**(before_epoch-15)
        else:
            lr = 0.1 * 1.1**12
            
    print("LR-schedule",lr_schedule_number,"is at", lr, "before epoch", before_epoch)
    return lr


def execute_training(modeltag, runs, autoencoder_stage, epoch, encoder_epoch, class_type, zero_center, verbose, dataset, learning_rate, learning_rate_decay, epsilon, lambda_comp, use_opti, encoder_version, options, ae_loss_name):
    #Get info like path of trainfile etc.
    dataset_info_dict = get_dataset_info(dataset)
    home_path=dataset_info_dict["home_path"]
    train_file=dataset_info_dict["train_file"]
    test_file=dataset_info_dict["test_file"]
    n_bins=dataset_info_dict["n_bins"]
    broken_simulations_mode=dataset_info_dict["broken_simulations_mode"] #def 0
    filesize_factor=dataset_info_dict["filesize_factor"]
    filesize_factor_test=dataset_info_dict["filesize_factor_test"]
    batchsize=dataset_info_dict["batchsize"] #def 32
    
    
    custom_objects=None
    #define loss function to use for new AEs
    print("Using AE loss:", ae_loss_name)
    if ae_loss_name=="mse":
        ae_loss="mse"
    elif ae_loss_name=="mae":
        ae_loss="mae"
    else:
        #custom loss functions have to be handed to load_model or it wont work
        custom_objects=get_custom_objects()
        ae_loss=custom_objects[ae_loss_name]
    
    
    
    #All models are now saved in their own folder   models/"modeltag"/
    model_folder = home_path + "models/" + modeltag + "/"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    #Optimizer used in all the networks:
    lr = learning_rate # 0.01 default for SGD, 0.001 for Adam
    #lr_decay can either be a float, e.g. 0.05 for 5% decay of lr per epoch,
    #or it can be a string like s1 for lr schedule 1. In this case, lr is ignored
    try:
        lr_decay=float(learning_rate_decay) # % decay for each epoch, e.g. if 0.05 -> lr_new = lr*(1-0.05)=0.95*lr
        lr_schedule_number=None # no schedule
    except ValueError:
        #then it is a schedule like s1 or some other string
        lr_schedule_number=learning_rate_decay
        lr_decay=0
        lr=0.001
        print("Using learning rate schedule", lr_schedule_number)
    
    
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
            lr=abs( lr * (1-float(lr_decay))**(epoch-1) )
            
        elif (autoencoder_stage==1 or autoencoder_stage==2 or autoencoder_stage==3)  and encoder_epoch>0:
            lr=abs( lr * (1-float(lr_decay))**(encoder_epoch-1) )
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
    train_tuple=[[train_file, int(h5_get_number_of_rows(train_file)*filesize_factor)]]
    test_tuple=[[test_file, int(h5_get_number_of_rows(test_file)*filesize_factor_test)]]
    
    
    #Check wheter a file with this name exists or not
    def check_for_file(proposed_filename):
        if(os.path.isfile(proposed_filename)):
            sys.exit(proposed_filename+ "exists already!")
    
    #Zero-Center with precalculated mean image
    #xs_mean = np.load(zero_center_file) if zero_center is True else None
    n_gpu=(1, 'avolkov')
    if zero_center == True:
        xs_mean = load_zero_center_data(train_files=train_tuple, batchsize=batchsize, n_bins=n_bins, n_gpu=n_gpu[0])
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
            model.compile(optimizer=adam, loss=ae_loss)
            #Create header for new test log file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_log_file:
                metrics = model.metrics_names #["loss"]
                test_log_file.write('{0}\tTest {1}\tTrain {2}\tTime\tLR'.format("Epoch", metrics[0], metrics[0]))
            
        else:
            #Load an existing trained autoencoder network and train that
            autoencoder_model_to_load=model_folder + "trained_" + modelname + '_epoch' + str(epoch) + '.h5'
            print("Loading existing autoencoder to continue training:", autoencoder_model_to_load)
            if lambda_comp==False:
                model = load_model(autoencoder_model_to_load, custom_objects=custom_objects)
            elif lambda_comp==True:
                #in case of lambda layers: Load model structure and insert weights, because load model is bugged for lambda layers
                print("Lambda mode enabled")
                model=setup_model(model_tag=modeltag, autoencoder_stage=0, modelpath_and_name=None, additional_options=options)
                model.load_weights(autoencoder_model_to_load)
                model.compile(optimizer=adam, loss=ae_loss)
                
                opti_weights=load_model(autoencoder_model_to_load, custom_objects=custom_objects).optimizer.get_weights()
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
            model = load_model(encoder_network_to_load, custom_objects=custom_objects)
    
    
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
            model = load_model(load_this, custom_objects=custom_objects)
    
    #Training of the supervised network on several different autoencoder epochs
    #epoch is calculated automatically and not used from user input
    #encoder epoch as usual
    #This does not use the same call for executing the training as stage 0,1 and 2
    
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    elif autoencoder_stage==3:
        #Always using custom lr schedule now, parser values are ignored
        #how many epochs should be trained on each autoencoder epoch, starting from epoch 1
        #if first epoch is 0, then the trained supervised network will be used
        if modeltag[:7] == "channel":
            #channel id autoencoders need less epochs per AE epoch, their modeltag starts with channel
            how_many_epochs_each_to_train =[1,]*100
        else:
            how_many_epochs_each_to_train =[10,]*1+[2,]*5+[1,]*194
        #model to initialize from if first epoch is 0
        #this one is only used for vgg_3_eps modeltag
        init_model_eps=model_folder + "trained_vgg_3_eps_autoencoder_epoch1_supervised_up_down_accdeg2_epoch26.h5"
        
        print("Autoencoder stage 3:\nParallel training with epoch schedule:", how_many_epochs_each_to_train[:20], ",...")
        
        def switch_encoder_weights(encoder_model, autoencoder_model, last_encoder_layer_index_override=None):
            #Change the weights of the frozen layers (up to the flatten layer) 
            #of the frozen encoder to that of another autoencoder model
            changed_layers=0
            #look for last encoder layer = last flatten layer in the network / layer with name encoded if present
            last_encoder_layer_index = 1
            if last_encoder_layer_index_override == None:
                for i,layer in enumerate(encoder_model.layers):
                    if layer.name == "encoded":
                        last_encoder_layer_index = i
                        break
                    elif layer.name == "after_encoded":
                        last_encoder_layer_index = i-1
                        break
                    elif "flatten" in layer.name:
                        last_encoder_layer_index = i
            else:
                last_encoder_layer_index = last_encoder_layer_index_override
            
            for i,layer in enumerate(encoder_model.layers):
                if i <= last_encoder_layer_index:
                    layer.set_weights(autoencoder_model.layers[i].get_weights())
                    changed_layers+=1
                else:
                    break
            print("Weights of layers changed:", changed_layers, "(up to layer", encoder_model.layers[last_encoder_layer_index].name, ")")
        
        #in case of the 200_dense model, manually set encoded layer (does not work otherwise...(it actually does...))
        if modeltag=="vgg_5_200_dense":
            last_encoder_layer_index_override=35
            print("Last encoder layer set to 35")
        else:
            last_encoder_layer_index_override=None
        
        #Encoder epochs after which to switch the autoencoder model
        switch_autoencoder_model=np.cumsum(how_many_epochs_each_to_train)
        #calculate the current autoencoder epoch automatically based on the encoder epoch
        #e.g. switch_at = [10,12,14], encoder_epoch = 11
        #--> AE epoch=2
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
                model_for_init = load_model(init_model, custom_objects=custom_objects)
                for i,layer in enumerate(model.layers):
                    layer.set_weights(model_for_init.layers[i].get_weights())
            
            
            #Create header for new test log file
            with open(model_folder + "trained_" + modelname + '_test.txt', 'w') as test_log_file:
                metrics = model.metrics_names #['loss', 'acc']
                test_log_file.write('{0}\tTest {1}\tTrain {2}\tTest {3}\tTrain {4}\tTime\tLR'.format("Epoch", metrics[0], metrics[0],metrics[1],metrics[1]))
            
        else:
            #Load an existing trained encoder network and train that
            model = load_model(model_folder + "trained_" + modelname + '_epoch' + str(encoder_epoch) + '.h5', custom_objects=custom_objects)
        
                
        #Own execution of training
        #Set LR of loaded model to new lr
        K.set_value(model.optimizer.lr, learning_rate)
            
        #Which epochs are the ones relevant for current stage
        running_epoch=encoder_epoch
            
        model.summary()
        print("Model: ", modelname)
        print("Current State of optimizer: \n", model.optimizer.get_config())
        filesize_hint="Filesize factor="+str(filesize_factor) if filesize_factor!=1 else ""
        filesize_hint_test="Filesize factor test="+str(filesize_factor_test) if filesize_factor_test!=1 else ""
        print("Train files:", train_tuple, filesize_hint)
        print("Test files:", test_tuple, filesize_hint_test)
        print("Using autoencoder model:", autoencoder_model)
        
        #Execute Training:
        for current_epoch in range(running_epoch,running_epoch+runs):
            #Does the model we are about to save exist already?
            check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
            #custom lr schedule; lr_decay was set to 0 already
            if lr_schedule_number != None:
                lr=lr_schedule(current_epoch+1, lr_schedule_number, learning_rate )
                K.set_value(model.optimizer.lr, lr)
            
            if current_epoch in switch_autoencoder_model:
                autoencoder_epoch+=1
                autoencoder_model = model_folder + "trained_" + modeltag + "_autoencoder_epoch" + str(autoencoder_epoch) + '.h5'
                print("Changing weights before epoch ",current_epoch+1," to ",autoencoder_model)
                switch_encoder_weights(model, load_model(autoencoder_model, custom_objects=custom_objects), last_encoder_layer_index_override)
            
            #Train network, write logfile, save network, evaluate network, save evaluation to file
            lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                                 batchsize=batchsize, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, epoch=current_epoch,
                                 shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                                 save_path=model_folder, is_autoencoder=is_autoencoder, verbose=verbose, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict)  
                
        sys.exit()
        
        
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
    #Which epochs are the ones relevant for current stage
    if is_autoencoder==True:
        running_epoch=epoch #Stage 0
    else:
        running_epoch=encoder_epoch #Stage 1 and 2
        
    #Set LR of loaded model to new lr
    if lr_schedule_number != None:
            lr=lr_schedule(running_epoch+1, lr_schedule_number, learning_rate )
    K.set_value(model.optimizer.lr, lr)
    
        
    model.summary()
    print("Model: ", modelname)
    print("Current State of optimizer: \n", model.optimizer.get_config())
    filesize_hint="Filesize factor="+str(filesize_factor) if filesize_factor!=1 else ""
    filesize_hint_test="Filesize factor test="+str(filesize_factor_test) if filesize_factor_test!=1 else ""
    print("Train files:", train_tuple, filesize_hint)
    print("Test files:", test_tuple, filesize_hint_test)
    if autoencoder_model is not None: print("Using autoencoder model:", autoencoder_model)
    
    #Execute Training:
    for current_epoch in range(running_epoch,running_epoch+runs):
        #This is before epoch current_epoch+1
        #Does the model we are about to save exist already?
        check_for_file(model_folder + "trained_" + modelname + '_epoch' + str(current_epoch+1) + '.h5')
        
        if lr_schedule_number != None:
            lr=lr_schedule(current_epoch+1, lr_schedule_number, learning_rate )
            K.set_value(model.optimizer.lr, lr)
            
        #print all the input for train_and_test_model for debugging
        """
        print("model=",model, "modelname=",modelname, "train_files=",train_tuple, "test_files=",test_tuple,
                             "batchsize=batchsize, n_bins=",n_bins, "class_type=",class_type, "xs_mean=",xs_mean, "epoch=",current_epoch,
                             "shuffle=False, lr=",lr, "lr_decay=",lr_decay, "tb_logger=False, swap_4d_channels=None,",
                             "save_path=",model_folder, "is_autoencoder=",is_autoencoder, "verbose=",verbose)
        """
        
        #Train network, write logfile, save network, evaluate network, save evaluation to file
        lr = train_and_test_model(model=model, modelname=modelname, train_files=train_tuple, test_files=test_tuple,
                             batchsize=batchsize, n_bins=n_bins, class_type=class_type, xs_mean=xs_mean, epoch=current_epoch,
                             shuffle=False, lr=lr, lr_decay=lr_decay, tb_logger=False, swap_4d_channels=None,
                             save_path=model_folder, is_autoencoder=is_autoencoder, verbose=verbose, broken_simulations_mode=broken_simulations_mode, dataset_info_dict=dataset_info_dict)    
    
    
if __name__ == "__main__":
    #the loss that is used for the autoencoder
    #usually "mse", "mae", or "mean_squared_error_poisson"
    ae_loss_name = "mse"
    
    execute_training(*unpack_parsed_args(), ae_loss_name)
