# -*- coding: utf-8 -*-

from keras.models import Model
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Input

def MaxUnpooling3D_Debug(Input_Tensor, Kernel_size=(2,2,2)):
    def MaxUnpooling3D_func(InputTensor, size):
        #in: b,o,n,m,c
        #out: b,2o,2n,2m,c
        #(batch_size, dim1, dim2, dim3, channels)
        batchsize=K.shape(InputTensor)[0]
        inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2], K.shape(InputTensor)[3])
        chan=K.shape(InputTensor)[4] #nchannels
        
        out=Input_Tensor
        #Add zeros to contracted dim
        padded=K.concatenate((out,K.zeros_like(out)), axis=4) #b,o,n,m,2*c
        #Reshape so that ncols is spaced with zeros
        out = K.reshape(padded, (batchsize,inshape[0],inshape[1],inshape[2]*size[2],chan)) #b,o,n,2*m,c

        return out
    
    Output_Tensor = Lambda(MaxUnpooling3D_func, arguments={"size":Kernel_size})(Input_Tensor)
    
    return Output_Tensor


inputs = Input(shape=(11,13,18,1))
x = MaxUnpooling3D_Debug(inputs,(1,1,2))
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer="adam", loss='mse')
model.save('test.h5')

