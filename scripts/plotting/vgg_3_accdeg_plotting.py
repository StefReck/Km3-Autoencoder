# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.text as mpl_text
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
import numpy as np

"""
Plot the loss of the vgg_3 autoencoders at certain epochs vs the accuracy of the frozen encoder
that was trained on that AE model at that epoch. The epoch of the E is annotated.
"""

#Up Down Classification
labels=[r'Autoencoder $\epsilon = 10^{-1}$', r'Autoencoder SGD', r'Autoencoder $\epsilon = 10^{-8}$',]
colors=["blue","green","red"]

#UP DOWN CLASSIFICATION
autoencoder_epochs = [[2,5,10,40,90,140],	
                      [2,10,50],
                      [1,2,5,10,60,112],]

losses = [[0.0753,	0.0719,	0.0704,	0.0679,	0.0489,	0.0359],	#basic eps 01
          [0.0678,0.0567, 0.0461], #sgd
          [0.0419, 0.0312,0.0258, 0.0230,	0.0187,	0.0183],] #eps08

accuracies=[[78.9,	81.6,	82.5,	82.1,	79.9,	78.6,],
            [82.0,80.6, 79.5],
            [77.3, 76.9, 76.3, 75.6,	76.4,	76.4,],]


#PARALELL TRAINING
#basic 
basic_parallel_logfile = "../Daten/trained_vgg_3_autoencoder_supervised_parallel_up_down_test.txt"
how_many_epochs_each_to_train =[5,]*10+[1,]*100

#eps
eps_parallel_logfile="../Daten/trained_vgg_3_eps_autoencoder_supervised_parallel_up_down_test.txt"
eps_how_many_epochs_each_to_train =[5,]*1+[2,]*100

#eps v2
eps_parallel_logfile_v2="../Daten/trained_vgg_3_eps_autoencoder_supervised_parallel_up_down_v2_test.txt"
eps_how_many_epochs_each_to_train_v2 =[5,]*10+[1,]*100

#MUON TO ELEC CC CLASSIFICATION
CCautoencoder_epochs = [[10,90],#basic e01
                        [],
                        [10,]] #eps08

CClosses = [[0.0704,	0.0489], #basic	
            [],
            [0.023,]] #eps08

CCaccuracies=[[59.9,	56.5],	
              [],
              [53.9,]]



def make_accdeg_plot(labels,colors, autoencoder_epochs,losses,accuracies,CCautoencoder_epochs,CClosses,CCaccuracies):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot([],[],marker="", ls="", label="Up-down classification")
    
    for i,autoencoder_model_loss in enumerate(losses):
        ax.plot(autoencoder_model_loss,accuracies[i], label=labels[i], color=colors[i], marker="o", ls="")
        for j,epoch in enumerate(autoencoder_epochs[i]):
            if i==1 and j==0:
                shiftx=-0.0015
                shifty=-0.2
            elif i==1:
                shiftx=-0.0022
                shifty=-0.2
            elif i==2 and j==5:
                shiftx=-0.0028
            else:
                shiftx=0.0005
                shifty=0
            ax.annotate(epoch, (losses[i][j]+shiftx,accuracies[i][j]+shifty), label="Epoch")
        
    plt.grid()
    
    #PID Auf zweiter axis plotten
    
    ax2 = ax.twinx()
    for i,autoencoder_model_loss in enumerate(CClosses):
        ax2.plot(autoencoder_model_loss,CCaccuracies[i],color=colors[i], marker="x", ls="", ms=7, mew=2)
        for j,epoch in enumerate(CCautoencoder_epochs[i]):
            shiftx=0.0005
            shifty=0
            #ax.annotate(epoch, (CClosses[i][j]+shiftx,CCaccuracies[i][j]+shifty))
            
    ax.plot([], [], '-', color='none', label=' ')
    ax.plot([],[],color="grey", marker="x", ls="", ms=7, mew=2, label="Particle ID")
    ax2.set_ylabel('Accuracy particle ID (%)')
    ax2.set_ylim((53,61))
    ax.set_ylim((75,83))
    ax.set_xlabel("Loss of autoencoder")
    
    """
    #PID auf leicher axis
    for i,autoencoder_model_loss in enumerate(CClosses):
        ax.plot(autoencoder_model_loss,CCaccuracies[i],color=colors[i], marker="x", ls="")
        for j,epoch in enumerate(CCautoencoder_epochs[i]):
            shiftx=0.0005
            shifty=0
            #ax.annotate(epoch, (CClosses[i][j]+shiftx,CCaccuracies[i][j]+shifty))
    """
     
     
        
    plt.title("Quality of autoencoders and encoders")
    plt.xlabel("Loss of autoencoder")
    ax.set_ylabel("Accuracy up-down (%)")
    
    handles, labels = ax.get_legend_handles_labels()
    
    
    class AnyObject(object):
        def __init__(self, text, color):
            self.my_text = text
            self.my_color = color
    
    class AnyObjectHandler(object):
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height
            patch = mpl_text.Text(x=11, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline', 
                                    horizontalalignment='center', multialignment=None, 
                                    fontproperties=None, linespacing=None, 
                                    rotation_mode=None)
            handlebox.add_artist(patch)
            return patch
    
    obj_0 = AnyObject(" 5", "black")
    
    
    legend=plt.legend(handles, labels)
    handler_map = legend.get_legend_handler_map()
    
    handler_map[obj_0] = AnyObjectHandler()
    
    legend=plt.legend(handles+[obj_0, ], labels+['Epoch of autoencoder', ],
               handler_map=handler_map, loc="lower right")
    
    
    plt.show()
    
    
def make_dict_from_file(test_file):
    with open(test_file, "r") as f:
        k = list(zip(*(line.strip().split('\t') for line in f)))
    data = {}
    for column in k:
        data[column[0]]=column[1:]
    return data

def transform_epoch(data,how_many_epochs_each_to_train):
    #returns the supervised epoch and best test acc of that AE model together with the corresponding autoencoder epoch
    epoch=np.array(data["Epoch"]).astype(int)
    acc=np.array(data["Test acc"]).astype(float)
    trainacc=np.array(data["Train acc"]).astype(float)
    take_acc_at_these_sup_epochs=np.cumsum(how_many_epochs_each_to_train)
    take=take_acc_at_these_sup_epochs[take_acc_at_these_sup_epochs<=max(epoch)]-1
    #      spvsd epoch, acc,       autoencoder epoch
    return epoch[take], acc[take], np.arange(1,len(acc[take])+1), trainacc[take]
        
def get_ae_loss_array(test_log):
    test_log_dict=make_dict_from_file(test_log)
    return np.array(test_log_dict["Test loss"]).astype(float)


def make_accdeg_parallel_basic(parallel_logfile,how_many_epochs_each_to_train, loss, accuracy, label, color):
    data=make_dict_from_file(parallel_logfile)
    trans_data=transform_epoch(data, how_many_epochs_each_to_train)
    ae_loss_array=get_ae_loss_array("../Daten/trained_vgg_3_autoencoder_test.txt")
    ae_loss_data = ae_loss_array[trans_data[2]-1]
    
    fig, ax = plt.subplots(figsize=(10,7))
    
    #plt.plot(data["Epoch"],data["Test acc"])
    test_plot=ax.plot(ae_loss_data, trans_data[1]*100, "o-", ms=4, color="orange", label="Test")
    ax.plot(ae_loss_data, trans_data[3]*100, "o--",alpha=0.5, lw=1, ms=4, label="Train",color=test_plot[0].get_color())
    ax.plot(loss, accuracy, "o", color=color, label=label)
    ax.grid()
    ax.legend()
    ax.set_xlabel("Loss of autoencoder")
    ax.set_ylabel("Accuracy up-down(%)")
    fig.suptitle("Parallel training of autoencoder and supervised network")
    xlims=[0.048,0.076]
    ax.set_xlim(xlims)
    ax.set_ylim([77,83])
    new_tick_locations_x = ae_loss_array[np.logical_and( ae_loss_array>xlims[0], ae_loss_array<xlims[1])]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations_x)
    ticklabels=[]
    for i in range(len(new_tick_locations_x)):
        i+=2
        if i in [2,5,10,20,40,50,60,70,80,85,90]:
            ticklabels.append(i)
        else:
            ticklabels.append("")
    ax2.set_xticklabels(ticklabels)
    ax2.set_xlabel("Autoencoder epoch")
    #ax2.set_aspect("equal")
    
    plt.show()
    

def make_accdeg_parallel_eps(parallel_logfile,how_many_epochs_each_to_train, loss, accuracy, label, color):
    data=make_dict_from_file(parallel_logfile)
    trans_data=transform_epoch(data, how_many_epochs_each_to_train)
    ae_loss_array=get_ae_loss_array("../Daten/trained_vgg_3_eps_autoencoder_test.txt")
    ae_loss_data = ae_loss_array[trans_data[2]-1]
    
    fig, ax = plt.subplots(figsize=(10,7))
    
    #plt.plot(data["Epoch"],data["Test acc"])
    test_plot=ax.plot(ae_loss_data, trans_data[1]*100, "o-", ms=4, color="orange", label="Test")
    ax.plot(ae_loss_data, trans_data[3]*100, "o--",alpha=0.5, lw=1, ms=4, label="Train",color=test_plot[0].get_color())
    ax.plot(loss, accuracy, "o", color=color, label=label)
    ax.grid()
    ax.legend()
    ax.set_xlabel("Loss of autoencoder")
    ax.set_ylabel("Accuracy up-down(%)")
    fig.suptitle("Parallel training of autoencoder and supervised network")
    xlims=[0.015,0.045]
    ax.set_xlim(xlims)
    new_tick_locations_x = ae_loss_array[np.logical_and( ae_loss_array>=xlims[0], ae_loss_array<=xlims[1])]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations_x)
    ticklabels=[]
    for i in range(len(new_tick_locations_x)):
        i+=1
        if i in [1,2,5,10,20,30,112]:
            ticklabels.append(i)
        else:
            ticklabels.append("")
    ax2.set_xticklabels(ticklabels)
    ax2.set_xlabel("Autoencoder epoch")
    #ax2.set_aspect("equal")
    
    plt.show()
    
make_accdeg_plot(labels,colors, autoencoder_epochs,losses,accuracies,CCautoencoder_epochs,CClosses,CCaccuracies)

#make_accdeg_parallel_basic(basic_parallel_logfile,how_many_epochs_each_to_train, losses[0], accuracies[0], labels[0], colors[0])
#make_accdeg_parallel_eps(eps_parallel_logfile,eps_how_many_epochs_each_to_train, losses[2], accuracies[2], labels[2], colors[2])
#make_accdeg_parallel_eps(eps_parallel_logfile_v2,eps_how_many_epochs_each_to_train_v2, losses[2], accuracies[2], labels[2], colors[2])
   
