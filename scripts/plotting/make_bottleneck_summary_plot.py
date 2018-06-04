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


mode="perf"

save_perf_plot_as="../../results/plots/statistics/bottleneck_summary_plot_updown_energy.pdf"
save_rob_plot_as= "../../results/plots/statistics/bottleneck_summary_plot_robust.pdf"
#size of bottleneck, Accuracy, Robuts1, Robuts2, Loss of energy
results = np.array([
[1920,	 82.51, 10.37, 5.72,   6.339 ], #"basic",
[600,    84.93, 9.26,	  5.09,   5.846 ],#"picture",
[200,    84.98, 7.47,	  4.18,   5.820 ],#"200",
[64,     84.71, 8.67,  6.59,   5.855  ],#"64 nodrop" für acc updown, aber rob up-down ist noch das alte	,
[32,     82.67,  8.08	,  5.85,  6.145 ],#"32-eps01 nodrop",
])

def make_and_save_plot(results, mode, save_perf_plot_as, save_rob_plot_as):
    figsize, fontsize = get_plot_statistics_plot_size("two_in_one_line")
    plt.rcParams.update({'font.size': fontsize})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if mode=="perf":
        #best performing unfrozen up/down and energy
        top_results=[88.16, 5.3185525]
        ax2 = ax.twinx()
       
        #ax2.axhline(top_results[1],0,1, ls="--", c="orange", lw=3, dashes=(3,3))
        #They are almost at the same heigth on different axis, hacky: plot them in one instead to line them up
        hline_acc = ax.axhline(top_results[0],0,1, ls="-", c="blue", lw=3)
        hline_loss = ax.axhline(top_results[0],0,1, ls="--", c="orange", lw=3, dashes=(3,3))
        
        acc_line, = ax.semilogx(results[:,0], results[:,1], "o", c="blue", label="Accuracy Up-Down")
        loss_line, = ax2.semilogx(results[:,0], results[:,4], "o", c="orange", label="MAE Energy")
        
        ax.set_ylabel("Accuracy up-down (%)")
        ax2.set_ylabel("Mean absolute error energy (GeV)")
        
        tick_locs=np.array([30,100,1000, 2000])
        ax.set_xticks(tick_locs.astype(int))
        ax.set_xticklabels(tick_locs.astype(str))
        ax.set_xlabel("Neurons in bottleneck")
        ax.set_xlim(28,2200)
        
        ax.set_ylim(80.2,88.5)
        ax2.set_ylim(6.39,5.27 )
        ax.yaxis.grid()
        
        plt.legend([(acc_line, loss_line)], ["Supervised"])
        
        ax2.legend(handles=[acc_line,loss_line,(hline_acc, hline_loss)],labels=["Accuracy Up-Down","MAE Energy","Supervised",], bbox_to_anchor=(1, 0.94), bbox_transform=ax.transAxes)
        plt.subplots_adjust(right=0.88)
        
        fig.savefig(save_perf_plot_as)
        print("Saved plot as", save_perf_plot_as)
        plt.show()
    
    elif mode=="rob":
        robust1_line, = ax.semilogx(results[:,0], results[:,2], "o", c="yellowgreen", label="Robust 1 Up-Down")
        robust2_line, = ax.semilogx(results[:,0], results[:,3], "o", c="darkolivegreen", label="Robust 2 Up-Down")
        
        ax.set_ylabel("Robustness")
        
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
        
        fig.savefig(save_rob_plot_as)
        print("Saved plot as", save_rob_plot_as)
        plt.show()

if mode=="both":
    make_and_save_plot(results, "perf", save_perf_plot_as, save_rob_plot_as)
    make_and_save_plot(results, "rob", save_perf_plot_as, save_rob_plot_as)
else:
    make_and_save_plot(results, mode, save_perf_plot_as, save_rob_plot_as)



