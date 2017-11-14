# -*- coding: utf-8 -*-

from keras.models import Model
from keras import backend as K
from keras.layers import Lambda, Dense, Flatten
from keras.layers import Input



def MaxUnpooling3D(Input_Tensor, Kernel_size=(2,2,2)):
    def MaxUnpooling3D_func(InputTensor, size):
        #in: b,o,n,m,c
        #out: b,2o,2n,2m,c
        #(batch_size, dim1, dim2, dim3, channels)
        batchsize=K.shape(InputTensor)[0]
        inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2], K.shape(InputTensor)[3])
        chan=K.shape(InputTensor)[4] #nchannels
        
        out=InputTensor
        
        if size[2]==2:
            #Add zeros to contracted dim
            padded=K.concatenate((out,K.zeros_like(out)), axis=4) #b,o,n,m,2*c
            #Reshape so that ncols is spaced with zeros
            out = K.reshape(padded, (batchsize,inshape[0],inshape[1],inshape[2]*size[2],chan)) #b,o,n,2*m,c
        
        if size[1]==2:
            #Space nrows the same way
            padded=K.concatenate((out,K.zeros_like(out)), axis=3) # b,n,2*2*m,c
            out=K.reshape(padded, (batchsize,inshape[0],inshape[1]*size[1],inshape[2]*size[2],chan)) # b,o,2*n,2*m,c
        
        if size[0]==2:
            #Space nrows the same way
            padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,o,2*2*n,2*m,c
            out=K.reshape(padded, (batchsize,inshape[0]*size[0],inshape[1]*size[1],inshape[2]*size[2],chan)) # b,2*o,2*n,2*m,c
        
        return out

    def MaxUnpooling3D_output_shape(input_shape):
        shape = list(input_shape)
        shape[1] *= Kernel_size[0]
        shape[2] *= Kernel_size[1]
        shape[3] *= Kernel_size[2]
        return tuple(shape)
    
    #Output_Tensor = Lambda(MaxUnpooling3D_func,MaxUnpooling3D_output_shape)(Input_Tensor)
    Output_Tensor = Lambda(MaxUnpooling3D_func, MaxUnpooling3D_output_shape, arguments={"size":Kernel_size})(Input_Tensor)
    return Output_Tensor


inputs = Input(shape=(1,1,1,1))
x = MaxUnpooling3D(inputs,(2,2,2))
x = Flatten()(x)
x = Dense(1, kernel_initializer='he_normal')(x)
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer="adam", loss='mse')
print(model.get_layer(index=3).get_weights())
model.save('test.h5')

del model
inputs = Input(shape=(1,1,1,1))
x = MaxUnpooling3D(inputs,(2,2,2))
x = Flatten()(x)
x = Dense(1, kernel_initializer='he_normal')(x)
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer="adam", loss='mse')
print(model.get_layer(index=3).get_weights())
model.load_weights("test.h5")
print(model.get_layer(index=3).get_weights())

"""
def MaxUnpooling3D_func(InputTensor, size):
    #in: b,o,n,m,c
    #out: b,2o,2n,2m,c
    #(batch_size, dim1, dim2, dim3, channels)
    batchsize=K.shape(InputTensor)[0]
    inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2], K.shape(InputTensor)[3])
    chan=K.shape(InputTensor)[4] #nchannels
    
    out=InputTensor
    
    if size[2]==2:
        #Add zeros to contracted dim
        padded=K.concatenate((out,K.zeros_like(out)), axis=4) #b,o,n,m,2*c
        #Reshape so that ncols is spaced with zeros
        out = K.reshape(padded, (batchsize,inshape[0],inshape[1],inshape[2]*size[2],chan)) #b,o,n,2*m,c
    
    if size[1]==2:
        #Space nrows the same way
        padded=K.concatenate((out,K.zeros_like(out)), axis=3) # b,n,2*2*m,c
        out=K.reshape(padded, (batchsize,inshape[0],inshape[1]*size[1],inshape[2]*size[2],chan)) # b,o,2*n,2*m,c
    
    if size[0]==2:
        #Space nrows the same way
        padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,o,2*2*n,2*m,c
        out=K.reshape(padded, (batchsize,inshape[0]*size[0],inshape[1]*size[1],inshape[2]*size[2],chan)) # b,2*o,2*n,2*m,c
    
    return out

Kernel_size=[2,2,2]

def MaxUnpooling3D_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= Kernel_size[0]
    shape[2] *= Kernel_size[1]
    shape[3] *= Kernel_size[2]
    return tuple(shape)

inputs = Input(shape=(11,13,18,1))
x = Lambda(MaxUnpooling3D_func, MaxUnpooling3D_output_shape, arguments={"size":Kernel_size})(inputs)
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer="adam", loss='mse')
model.save('test.h5')
"""


