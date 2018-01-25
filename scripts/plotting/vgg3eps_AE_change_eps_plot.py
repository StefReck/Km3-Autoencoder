# -*- coding: utf-8 -*-
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


data_for_plots=np.load("../Daten/vgg_3_eps_AE_loss.npy",encoding='latin1')
#[epoch, test value, train value, name, train epoch, ylabel]

plot_in_range_array=[[114,200],[95,113]]
colors_array=["blue","red"]
label_array=[r"Adam with $\epsilon = 10^{-8}$",r"Adam with $\epsilon = 10^{-1}$"]

fig, ax=plt.subplots(figsize=(8,6))
for rangeno,plot_in_range in enumerate(plot_in_range_array):
    for i,data_of_model in enumerate(data_for_plots):
        for j,arr in enumerate(data_of_model):
            if j in [0,1,2,4]:
                data_of_model[j]=np.array(arr).astype(float)
        test_which=np.logical_and(data_of_model[0]>=plot_in_range[0]-1, data_of_model[0]<=plot_in_range[1])
        train_which=np.logical_and(data_of_model[4]>=plot_in_range[0]-1, data_of_model[4]<=plot_in_range[1])
        
        test_E=data_of_model[0][test_which]
        test_loss=data_of_model[1][test_which]
        train_E=data_of_model[2][train_which]
        train_loss=data_of_model[4][train_which]
        
        test_plot = ax.plot(test_E, test_loss, marker="o", color=colors_array[rangeno])
        ax.plot([],[],color=colors_array[rangeno],lw=3, label=label_array[rangeno])
        ax.plot(train_loss, train_E, linestyle="-", color=test_plot[0].get_color(), alpha=0.5, lw=0.6)
        #handle_for_legend = mlines.Line2D([], [], color=test_plot[0].get_color(), lw=3, label=label_array[i])
        #handles1.append(handle_for_legend)


plt.xticks( np.arange(90,120,1) )
[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
plt.xlim((94.5,117.5))
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax.set_title("Change to the epsilon parameter during autoencoder training", fontsize=15)

lhandles, llabels = ax.get_legend_handles_labels()
legend1 = plt.legend(handles=lhandles, labels=llabels, loc="upper right")

test_line = mlines.Line2D([], [], color='grey', marker="o", label='Test')
train_line = mlines.Line2D([], [], color='grey', linestyle="-", alpha=0.5, lw=2, label='Train')
legend2 = plt.legend(handles=[test_line,train_line], loc="upper left")

ax.add_artist(legend1)
ax.add_artist(legend2)

plt.grid(True)
fig.tight_layout()
plt.show()
