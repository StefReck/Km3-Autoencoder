# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

#Test Files to make plots from; Format: vgg_3/trained_vgg_3_autoencoder_test.txt
test_files = ["vgg_3/trained_vgg_3_autoencoder_test.txt", 
              "vgg_3_dropout/trained_vgg_3_dropout_autoencoder_test.txt",
              "vgg_3_max/trained_vgg_3_max_autoencoder_test.txt",
              "vgg_3_stride/trained_vgg_3_stride_autoencoder_test.txt"]

#For debugging
#test_files = ["trained_vgg_3_autoencoder_test.txt", "trained_vgg_3_dropout_autoencoder_test.txt"]

modelnames=[] #e.g. vgg_3_autoencoder
for modelident in test_files:
    modelnames.append(modelident.split("trained_")[1][:-9])
    
def make_dicts_from_files(test_files):
    dict_array=[]
    for test_file in test_files:
        with open("models/"+test_file) as f:
            k = list(zip(*(line.strip().split('\t') for line in f)))
        data = {}
        for column in k:
            data[column[0]]=column[1:]
        dict_array.append(data)
    return dict_array


def make_test_train_plot(epochs, test, train, label):
    test_plot = plt.plot(epochs, test, label=label)
    plt.plot(epochs, train, linestyle="--", color=test_plot[0].get_color(), label="")
    
    #x_ticks_major = np.arange(0, 101, 10)
    #plt.xticks(x_ticks_major)
    #plt.minorticks_on()
    #plt.ylim((0, 1))
   

dict_array = make_dicts_from_files(test_files)

for i,data_dict in enumerate(dict_array):
    make_test_train_plot(data_dict["Epoch"], data_dict["Test loss"], data_dict["Train loss"], modelnames[i])
    
plt.legend()
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.title("Progress")
plt.grid(True)
plt.show()