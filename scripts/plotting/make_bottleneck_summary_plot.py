# -*- coding: utf-8 -*-
"""
Plot the accuracy up-down and the loss energy over the training set as a function 
of the bottleneck size for the best models of each size.
"""
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../util/')
from saved_setups_for_plot_statistics import get_plot_statistics_plot_size

#perf, rob
mode="perf"

save_perf_plot_as="../../results/plots/statistics/bottleneck_summary_plot_updown_energy.pdf"
save_rob_4_plot_as= "../../results/plots/statistics/bottleneck_summary_plot_robust_broken4.pdf"
save_rob_14_plot_as= "../../results/plots/statistics/bottleneck_summary_plot_robust_broken14.pdf"
save_names=[save_perf_plot_as, save_rob_4_plot_as, save_rob_14_plot_as]

#size of bottleneck, 
#          acc
#               Loss of energy
#                       MRE energy

results = np.array([
[1920,	 82.51, 6.339, 0.28014   ], #"basic",
[600,    84.93, 5.846, 0.26500  ],#"picture",
[200,    84.98, 5.774, 0.26084  ],#"200/200-dense",
[64,     84.71, 5.838, 0.26244  ],#"64 nodrop"
[32,     82.67, 6.145, 0.28251 ],#"32-eps01 nodrop",
])

#Broken4 updown
robust_4 = np.array([
[1920,  44.28,	35.82],#basic unfrozen
[1920,	 10.37, 5.72, ], #"basic",
[600,    9.26,	  5.09,  ],#"picture",
[200,    7.47,	  4.18, ],#"200",
[64,     8.40,	4.72   ],#"64 nodrop"
[32,     8.08	,  5.85, ],#"32-eps01 nodrop",
])

#Broken 14 energy robustness 1 and 2
robustness = np.array([
[1920,  53.3354561,     28.70117248], #basic unfrozen
[1920,  40.32351996	,    12.56009858],#basic enc
[600,   37.91563508,    11.86464918],#600 picture enc
#[200,   29.6020633,	     7.915001627],#200 enc
[200,   25.41014215, 	  6.329089502],#200_dense
[64,    7.259040325	,   -0.385445984],#64
[32,    9.77844588, 	  1.001811581],#32
])



data=[results, robust_4, robustness]

def make_and_save_plot(data, mode, save_names):
    results = data[0]
    robust_4 = data[1]
    robustness = data[2] #energy broken14
    
    robust1_color="yellowgreen"
    robust2_color="darkolivegreen"
    
    figsize, fontsize = get_plot_statistics_plot_size("two_in_one_line")
    plt.rcParams.update({'font.size': fontsize})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if mode=="perf":
        #best performing unfrozen up/down and energy
        top_results=[88.16, 5.3185525, 24.7034]
        ax2 = ax.twinx()
        make_MAE=0
        acc_data=[results[:,0], results[:,1]]
        if make_MAE:
            ergy_data=[results[:,0], results[:,2]]
            y_label="MAE Energy"
            y_label_ax = "Mean absolute error energy (GeV)"
            hline_loc=top_results[1]
            ylims=[6.39,5.27]
            legend_label="MAE Energy"
        else:
            #make MRE instead
            ergy_data=[results[:,0], 100*results[:,3]]
            y_label="MRE Energy"
            y_label_ax = "Mean relative error energy (%)"
            hline_loc=top_results[2]
            ylims=[28.5,24.56]
            legend_label="MRE Energy"
        #ax2.axhline(top_results[1],0,1, ls="--", c="orange", lw=3, dashes=(3,3))
        #They are almost at the same heigth on different axis, hacky: plot them in one instead to line them up
        hline_acc = ax.axhline(top_results[0],0,1, ls="-", c="blue", lw=3)
        hline_loss = ax.axhline(top_results[0],0,1, ls="--", c="orange", lw=3, dashes=(3,3))
        
        acc_line, = ax.semilogx(acc_data[0], acc_data[1], "o", c="blue", label="Accuracy Up-Down")
        loss_line, = ax2.semilogx(ergy_data[0], ergy_data[1], "o", c="orange", label=y_label)
        
        ax.set_ylabel("Accuracy up-down (%)")
        ax2.set_ylabel(y_label_ax)
        
        tick_locs=np.array([30,100,1000, 2000])
        ax.set_xticks(tick_locs.astype(int))
        ax.set_xticklabels(tick_locs.astype(str))
        ax.set_xlabel("Neurons in bottleneck")
        ax.set_xlim(28,2200)
        
        ax.set_ylim(80.2,88.5)
        ax2.set_ylim(ylims)
        ax.yaxis.grid()
        
        plt.legend([(acc_line, loss_line)], ["Supervised"])
        
        ax2.legend(handles=[acc_line,loss_line,(hline_acc, hline_loss)],labels=["Accuracy Up-Down",legend_label,"Supervised",], bbox_to_anchor=(1, 0.94), bbox_transform=ax.transAxes)
        plt.subplots_adjust(right=0.88)
        
        fig.savefig(save_names[0])
        print("Saved plot as", save_names[0])
        plt.show()
    
    elif mode=="rob":
        plot_data = robust_4
        
        robust1_line, = ax.semilogx(plot_data[1:,0], plot_data[1:,1], "o--", c=robust1_color, label="Overestimation")
        robust2_line, = ax.semilogx(plot_data[1:,0], plot_data[1:,2], "o--", c=robust2_color, label="Underperformance")
        
        #hline_acc = ax.axhline(plot_data[0][1],0,1, ls="-", c=robust1_color, )
        #hline_loss = ax.axhline(plot_data[0][2],0,1, ls="-", c=robust2_color, )
        #plt.plot([32,],[32,], color="grey", ls="-", label="Supervised")
        
        ax.set_ylabel("Percentage")
        
        #ax.set_ylim(80.2,88.5)
        #ax.yaxis.grid()
        plt.subplots_adjust(right=0.88)
        plt.legend()
        plt.grid()
        
        tick_locs=np.array([30,100,1000, 2000])
        ax.set_xticks(tick_locs.astype(int))
        ax.set_xticklabels(tick_locs.astype(str))
        ax.set_xlabel("Neurons in bottleneck")
        ax.set_xlim(28,2200)
        
        fig.savefig(save_names[1])
        print("Saved plot as", save_names[1])
        plt.show()
    
    elif mode=="rob2":
        plot_data = robustness
        
        robust1_line, = ax.semilogx(plot_data[1:,0], plot_data[1:,1], "o--", c=robust1_color, label="Overestimation")
        robust2_line, = ax.semilogx(plot_data[1:,0], plot_data[1:,2], "o--", c=robust2_color, label="Underperformance")
        
        hline_acc = ax.axhline(plot_data[0][1],0,1, ls="-", c=robust1_color, )
        hline_loss = ax.axhline(plot_data[0][2],0,1, ls="-", c=robust2_color, )
        plt.plot([32,],[32,], color="grey", ls="-", label="Supervised")
        
        ax.set_ylabel("Percentage")
        
        #ax.set_ylim(80.2,88.5)
        #ax.yaxis.grid()
        plt.subplots_adjust(right=0.88)
        plt.legend()
        plt.grid()
        
        tick_locs=np.array([30,100,1000, 2000])
        ax.set_xticks(tick_locs.astype(int))
        ax.set_xticklabels(tick_locs.astype(str))
        ax.set_xlabel("Neurons in bottleneck")
        ax.set_xlim(28,2200)
        
        fig.savefig(save_names[2])
        print("Saved plot as", save_names[2])
        plt.show()
        

if mode=="all":
    make_and_save_plot(data, "perf", save_names)
    make_and_save_plot(data, "rob", save_names)
    make_and_save_plot(data, "rob2", save_names)
else:
    make_and_save_plot(data, mode, save_names)



