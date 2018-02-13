# -*- coding: utf-8 -*-
"""
Small script to plot Autoencoder loss history and the accuracy
history of its parallel training.
Erstellt wurde hiermit:
vgg_5_picture_parallel_loss_and_acc.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

file_AE="../Daten/vgg5_picture_AE_prl_plotdata_AE.npy"
file_prl="../Daten/vgg5_picture_AE_prl_plotdata_prl.npy"

#[epoch, test value, train value, name, train epoch, ylabel]
AE_data = np.load(file_AE, encoding='latin1')[0]

prl_data = np.load(file_prl, encoding='latin1')[0][[0,1,4,2]]
how_many_epochs_each_to_train = np.array([10,]*1+[2,]*5+[1,]*64)
take_these_epochs=np.cumsum(how_many_epochs_each_to_train)

for i in range(len(prl_data[0:2])):
    prl_data[i]=np.array(prl_data[i])[take_these_epochs-1]

def plot_test_train(testE, testloss, trainE, trainloss, label, ax):
    test_plot = ax.plot(testE, testloss, marker="o", ms=3, label=label)
    train_plot= ax.plot(trainE, trainloss, alpha=0.5, lw=0.6, linestyle="-", color=test_plot[0].get_color())


fig,ax = plt.subplots(figsize=(10,7))
plt.grid()

plot_test_train(AE_data[0],AE_data[1],AE_data[4],AE_data[2], "Autoencoder loss", ax)

ax2 = ax.twinx()
ax2.plot(AE_data[0][:len(prl_data[0])],prl_data[1], marker="o", ms=3, color="orange", label="Encoder accuracy")

#plot_test_train(prl_data[0],prl_data[1],prl_data[2],prl_data[3],ax2)
ax.set_ylim([0.05,0.09])
ax.set_xlabel("Autoencoder epoch")
ax.set_ylabel("Loss")
ax.text(x=38,y=0.066,s=r'$\epsilon = 10^{-1}$')

ax2.set_ylabel("Accuracy")

handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

plt.legend(handles=handles+handles2, labels=labels+labels2)
plt.title("Autoencoder loss and encoder accuracy during parallel training")

plt.show()
