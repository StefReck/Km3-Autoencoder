# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.text as mpl_text

"""
Plot the loss of the vgg_3 autoencoders at certain epochs vs the accuracy of the frozen encoder
that was trained on that AE model at that epoch. The epoch of the E is annotated.
"""


labels=[r'Autoencoder $\epsilon = 10^{-1}$', r'Autoencoder $\epsilon = 10^{-8}$', r'Autoencoder SGD']
autoencoder_epochs = [ [2,5,10,40,90,140],	
          [2,10,60,112],	
          [2,10,50] ]

losses = [ [0.0753,	0.0719,	0.0704,	0.0679,	0.0489,	0.0359],	
          [0.0312, 0.0230,	0.0187,	0.0183],	
          [0.0678,0.0567, 0.0461] ]

accuracies=[[78.9,	81.6,	82.5,	82.1,	79.9,	78.6,],
            [76.7, 74.5,	75.5,	76.4,],
            [82.0,80.6, 79.5]]

#accuracy taken at epoch x[0] from x[1] trained in total
at_encoder_epoch= [ [[17,26],	[16,60],[23,52],[7,26],[56, 102],[45,77],],
                    [[4,26],[14,15],[54,77],],
                    [[14,32],]  ]


fig, ax = plt.subplots()

for i,autoencoder_model_loss in enumerate(losses):
    plt.plot(autoencoder_model_loss,accuracies[i], label=labels[i], marker="o", ls="")
    for j,epoch in enumerate(autoencoder_epochs[i]):
        if i==2 and j==0:
            shiftx=-0.0015
            shifty=-0.2
        else:
            shiftx=0.0005
            shifty=0
            
        ax.annotate(epoch, (losses[i][j]+shiftx,accuracies[i][j]+shifty), label="Epoch")
    
plt.title("Quality of autoencoders end encoders (up-down classification)")
plt.xlabel("Loss of autoencoder")
plt.ylabel("Accuracy of frozen encoder (%)")
plt.grid()

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

obj_0 = AnyObject("5", "black")


legend=plt.legend(handles, labels)
handler_map = legend.get_legend_handler_map()

handler_map[obj_0] = AnyObjectHandler()

legend=plt.legend(handles+[obj_0, ], labels+['Epoch of autoencoder', ],
           handler_map=handler_map, loc="lower right")


plt.show()

