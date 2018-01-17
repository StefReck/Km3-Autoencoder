# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.text as mpl_text
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

"""
Plot the loss of the vgg_3 autoencoders at certain epochs vs the accuracy of the frozen encoder
that was trained on that AE model at that epoch. The epoch of the E is annotated.
"""

#Up Down Classification
labels=[r'Autoencoder $\epsilon = 10^{-1}$', r'Autoencoder SGD', r'Autoencoder $\epsilon = 10^{-8}$',]
colors=["blue","green","red"]
autoencoder_epochs = [[2,5,10,40,90,140],	
                      [2,10,50],
                      [1,2,10,60,112],]

losses = [[0.0753,	0.0719,	0.0704,	0.0679,	0.0489,	0.0359],	#basic eps 01
          [0.0678,0.0567, 0.0461], #sgd
          [0.0419, 0.0312, 0.0230,	0.0187,	0.0183],] #eps08

accuracies=[[78.9,	81.6,	82.5,	82.1,	79.9,	78.6,],
            [82.0,80.6, 79.5],
            [67.5, 76.9, 74.5,	75.5,	76.4,],]

#accuracy taken at epoch x[0] from x[1] trained in total
at_encoder_epoch= [ [[17,26],	[16,60],[23,52],[7,26],[56, 102],[45,77],],
                    [[4,26],[14,15],[54,77],],
                    [[14,32],]  ]

#muon to elec CC
CCautoencoder_epochs = [[10,90],#basic e01
                        [],
                        [10,]] #eps08

CClosses = [[0.0704,	0.0489], #basic	
            [],
            [0.023,]] #eps08

CCaccuracies=[[59.9,	56.5],	
              [],
              [53.9,]]



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
ax2.set_ylim((45,61))
ax.set_ylim((67,83))
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

