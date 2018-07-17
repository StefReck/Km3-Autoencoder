# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Conv3D, Conv3DTranspose, AveragePooling3D, UpSampling3D, Dense, Flatten, Reshape
import numpy as np

from model_definitions import setup_model

tex_file_path="../results/tex_tables/tables_of_models.tex"
models_to_print=["vgg_3", 
                 "vgg_4_6c", "vgg_4_6c_scale",
                 "vgg_4_8c", "vgg_4_10c",
                 "vgg_4_15c", 
                 #"vgg_4_30c",
                 "vgg_5_picture", "vgg_5_channel", "vgg_5_morefilter",
                 "vgg_5_200", "vgg_5_200_dense",
                 "vgg_5_64", "vgg_5_32",
                 "vgg_5_200_large","vgg_5_200_deep",
                 "vgg_5_200_small", "vgg_5_200_shallow",
                 "channel_5n", "channel_3n_m3",]

#Placeholder captions:
captions = ["Network structure of the "+modeltag.replace("_"," ")+" model." for modeltag in models_to_print]




def print_model_blockwise_latex_ready(model, texfile):
    #The content of each tabel
    print_these_layers = [Conv3DTranspose, Conv3D, 
                          AveragePooling3D, UpSampling3D,
                          Dense, Flatten, Reshape]
    names_of_these_layers = ["\\phantom{(T)} Convolutional block (T)", "Convolutional block",
                             "Average Pooling", "Upsampling",
                             "Dense Block", "Flatten", "Reshape"]
    #Input layer is treated seperatly!
    
    #Generate a list of the above layers, together with their output dimension
    layers_list = []
    bottleneck_is_at_layer=0
    bottleneck_neurons=1000000
    for layer_no, layer in enumerate(model.layers):
        for i, layer_type in enumerate(print_these_layers):
            if layer_no==0 or isinstance(layer, layer_type):
                if layer_no==0:
                    name="Input"
                else:
                    name = names_of_these_layers[i]
                out_size=layer.output_shape[1:]
                layers_list.append([name,out_size])
                
                if np.prod(out_size) < bottleneck_neurons:
                    bottleneck_neurons = np.prod(out_size)
                    bottleneck_is_at_layer = len(layers_list)-1
                break
            
    print("    Bottleneck:", bottleneck_neurons, "neurons")
    #Print a line in the table for each of the above layers
    for layer_no, [name,out_size] in enumerate(layers_list):
        is_at_bottleneck = (layer_no==bottleneck_is_at_layer)
        is_last_layer = (layer_no==len(layers_list)-1)
        make_appendix=True
        
        if len(out_size)!=4:
            if is_at_bottleneck:
                dimensions = "\\multicolumn{3}{c|}{"+str(out_size[0])+"} \\\\ \\hline \\hline"
                make_appendix=False
            else:
                dimensions = "\\multicolumn{3}{c|}{"+str(out_size[0])+"}"
        else:
            dimensions = str(out_size[0])+"x"+str(out_size[1])+"x"+str(out_size[2])+"&x&"+str(out_size[3])
        
        prefix=""
        if layer_no==0 or layer_no==len(layers_list)-1:
            prefix="\\rowcolor{Gray} "
            
        if is_last_layer:
            if name=="Convolutional block":
                name = "Convolution (kernel size 1, linear)"
            elif name=="Dense Block":
                name = "Dense (linear)"
        
        if make_appendix:
            if is_at_bottleneck:
                appendix = " ("+str(bottleneck_neurons)+") \\\\ \\hline"
            else:
                appendix = " \\\\"
            try:
                if name=="Average Pooling" or name=="Input" or layers_list[layer_no+1][0]=="Upsampling":
                    appendix+=" \\hline\n"
            except IndexError:
                pass
        else:
            appendix=""
        
        if name=="Average Pooling":
            out_size_previous=layers_list[layer_no-1][1]
            factors=[]
            for i in range(len(out_size)):
                factors.append(str(int(round(out_size_previous[i]/out_size[i]))))
            name+=" ("+factors[0]+","+factors[1]+","+factors[2]+")"
            
        if name=="Upsampling":
            out_size_previous=layers_list[layer_no-1][1]
            factors=[]
            for i in range(len(out_size)):
                factors.append(str(int(round(out_size[i]/out_size_previous[i]))))
            name+=" ("+factors[0]+","+factors[1]+","+factors[2]+")"
            
        if is_last_layer or layer_no==len(layers_list)-2: 
            appendix += " \\hline"
        texfile.write(prefix+name+" & "+dimensions+appendix+"\n") 

def print_table_of_model(modeltag, texfile, caption):
    #complete table of a model
    model = setup_model(modeltag, autoencoder_stage=0)
    trainable_weights = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    
    #command_name="table"+modeltag.replace("_", "")
    print("Generating table for model", modeltag)
    texfile.write("\n\n%model "+modeltag+"\n")
    #texfile.write("\\newcommand{\\"+command_name+"}{"+"\n")
    texfile.write("\\begin{table}"+"\n")
    texfile.write("\\caption{"+caption+"}"+"\n")
    texfile.write("\\vspace*{5pt}"+"\n")
    texfile.write("\\centering"+"\n")
    texfile.write("\\setlength{\\extrarowheight}{0.1cm}"+"\n")
    texfile.write("\\setlength{\\tabcolsep}{0.4cm}"+"\n")
    texfile.write("\\begin{tabular}{|c|rcl|}"+"\n")
    texfile.write("\\multicolumn{1}{c}{Building block} & \\multicolumn{3}{c}{Output dimension} \\\\ "+"\n")
    texfile.write("\\hline"+"\n")
    
    print_model_blockwise_latex_ready(model, texfile)

    texfile.write("\\end{tabular}"+"\n")
    texfile.write("\\begin{center} Trainable parameters: "+str(trainable_weights)+" \\end{center}")
    texfile.write("\\label{tab_"+modeltag+"}"+"\n")
    texfile.write("\\end{table}"+"\n\n")

def make_tex_file(tex_file_path, models_to_print, captions):
    with open(tex_file_path, 'w') as texfile:
        texfile.write("\\input{header.tex}"+"\n")
        texfile.write("\\usepackage[utf8]{inputenc}"+"\n")
        texfile.write("\\usepackage{color, colortbl}"+"\n")
        texfile.write("\\usepackage{xspace}"+"\n")
        texfile.write("\\definecolor{Gray}{gray}{0.9}"+"\n")
        texfile.write("\\definecolor{LightRed}{rgb}{1, 0.92, 0.92}"+"\n")
        texfile.write("\\newcommand{\\model}[1]{\\textsc{#1}\\xspace}"+"\n")
        texfile.write("\\begin{document}"+"\n")
        
        
        for modelno, modeltag in enumerate(models_to_print):
            print_table_of_model(modeltag, texfile, captions[modelno])

        texfile.write("\\end{document}")
        
make_tex_file(tex_file_path, models_to_print, captions)
#make_tex_file(tex_file_path, ["channel_vgg",], ["test",])



