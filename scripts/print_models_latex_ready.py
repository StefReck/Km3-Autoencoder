# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Conv3D, Conv3DTranspose, AveragePooling3D, UpSampling3D, Dense, Flatten, Reshape
import numpy as np

from model_definitions import setup_model

tex_file_path="../results/tex_tables/tables_of_models.tex"
models_to_print=[ #modeltag, display name
                 #["vgg_3", "model-1920"],
                 #"vgg_4_6c", 
                 "\\section{For chapter 4}",
                 ["vgg_4_6c_scale", "model-1920 12 layers"],
                 ["vgg_4_8c", "model-1920 16 layers"], 
                 ["vgg_4_10c", "model-1920 20 layers"],
                 ["vgg_4_15c", "model-1920 30 layers"], 
                 #["vgg_4_30c", "model-1920 60 layers"],
                 ["vgg_4_10c_smallkernel", "small kernel"],
                 ["vgg_4_10c_triple", "triple"],
                 ["vgg_4_10c_triple_same_structure", "triple variation"],
                 
                 "\\clearpage \n \\newpage \n \\section{For chapter 5} ",
                 ["vgg_5_picture", "model-600 picture"], 
                 ["vgg_5_channel", "model-600 filter"], 
                 ["vgg_5_morefilter", "model-600 more filter"],
                 ["vgg_5_200", "model-200"], 
                 ["vgg_5_200_dense", "model-200 dense"],
                 ["vgg_5_64", "model-64"], 
                 ["vgg_5_32", "model-32"],
                 ["vgg_5_200_large", "model-200 wide"],
                 ["vgg_5_200_deep", "model-200 deep"],
                 ["vgg_5_200_small", "model-200 small"], 
                 ["vgg_5_200_shallow", "model-200 shallow"],
                 
                 "\\clearpage \n \\newpage \n \\section{For chapter 6}",
                 ["channel_5n", "channel-5"], 
                 ["channel_3n_m3", "channel-3"],
                ]
captions, labels = [], []
for models_to_print_line in models_to_print:
    if type(models_to_print_line)==str:
        caption, label = "", ""
    else:
        modeltag, display_name = models_to_print_line
        
        caption = "Network architecture of the \\model{"+display_name+"} autoencoder." 
        if modeltag == "vgg_4_10c_smallkernel":
            caption += " The standard kernel size for convolutions and \
            transposed convolutions is $2\\times 2\\times 2$ for this model."
        
        label = "app_"+display_name.replace(" ","_")
    captions.append(caption)
    labels.append(label)


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

def print_table_of_model(modeltag, texfile, caption, label):
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
    #texfile.write("\\setlength{\\extrarowheight}{0.1cm}"+"\n")
    texfile.write("\\setlength{\\tabcolsep}{0.4cm}"+"\n")
    texfile.write("\\begin{tabular}{|c|rcl|}"+"\n")
    texfile.write("\\multicolumn{1}{c}{Building block} & \\multicolumn{3}{c}{Output dimension} \\\\ "+"\n")
    texfile.write("\\hline"+"\n")
    
    print_model_blockwise_latex_ready(model, texfile)

    texfile.write("\\end{tabular}"+"\n")
    texfile.write("\\begin{center} Trainable parameters: "+str(trainable_weights)+" \\end{center}")
    texfile.write("\\label{"+label+"}"+"\n")
    texfile.write("\\end{table}"+"\n\n")

def make_tex_file(tex_file_path, models_to_print, captions, labels):
    with open(tex_file_path, 'w') as texfile:
        #write a header:
        """
        texfile.write("\\input{header.tex}"+"\n")
        texfile.write("\\usepackage[utf8]{inputenc}"+"\n")
        texfile.write("\\usepackage{color, colortbl}"+"\n")
        texfile.write("\\usepackage{xspace}"+"\n")
        texfile.write("\\definecolor{Gray}{gray}{0.9}"+"\n")
        texfile.write("\\definecolor{LightRed}{rgb}{1, 0.92, 0.92}"+"\n")
        texfile.write("\\newcommand{\\model}[1]{\\textsc{#1}\\xspace}"+"\n")
        texfile.write("\\begin{document}"+"\n")
        """
        
        for modelno, models_to_print_line in enumerate(models_to_print):
            if type(models_to_print_line)==str:
                texfile.write(models_to_print_line+"\n")
                continue
            [modeltag, __] = models_to_print_line
            print_table_of_model(modeltag, texfile, captions[modelno], labels[modelno])

        #texfile.write("\\end{document}")
    print("Done! Texfile saved to", tex_file_path)
    
make_tex_file(tex_file_path, models_to_print, captions, labels)
#make_tex_file(tex_file_path, ["channel_vgg",], ["test",])



