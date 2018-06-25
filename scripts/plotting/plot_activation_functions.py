# -*- coding: utf-8 -*-
"""
Make plot of activation functions.
"""

import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    #for 2 inputs, one of which is 0
    return np.exp(x)/(np.exp(x)+np.exp(0))

figsize = [6.4,5.5]   
font_size=14

plt.rcParams.update({'font.size': font_size})
plt.subplots(figsize=figsize)


xrange=(-6,6)
x = np.linspace(xrange[0],xrange[1],1000)

plt.plot(x,softmax(x), label="Softmax")
plt.plot(x, (0.5*np.tanh(x)+0.5), label="tanh")
plt.plot(x, np.maximum(0,x), label="ReLu")


plt.xlim(xrange)
plt.ylim(-0.1,1.1)

plt.xlabel(r"x")
plt.ylabel(r"f(x)")
plt.grid()

plt.legend()
plt.show()

save_as="../../results/plots/activation_functions.pdf"
plt.savefig(save_as)
print("Saved plot to", save_as)
