# -*- coding: utf-8 -*-


"""
Contains the definition for MaxUnpooling2D and 3D Layers. Only 2x2 for 2D
and Kernels with i,j,k = 1 or 2 are supported
"""


from keras import backend as K
from keras.layers import Lambda


def MaxUnpooling2D(Input_Tensor):
    def MaxUnpooling2D_func(InputTensor):
        #in: b,n,m,c
        #out: b,2n,2m,c
        #(batch_size, rows, cols, channels)
        batchsize=K.shape(InputTensor)[0]
        inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2]) #(nrows,ncols)
        chan=K.shape(InputTensor)[3] #nchannels
        
        #Add zeros to dimension next to ncols that will be spaced (which is channels here) 
        padded=K.concatenate((InputTensor,K.zeros_like(InputTensor)), axis=3) #b,n,m,2*c
        #Reshape so that ncols is spaced with zeros
        out = K.reshape(padded, (batchsize, inshape[0],inshape[1]*2,chan)) #b,n,2*m,c
        
        #Space nrows the same way
        padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,n,2*2*m,c
        out=K.reshape(padded, (batchsize, inshape[0]*2,inshape[1]*2,chan)) # b,2*n,2*m,c
        
        return out

    def MaxUnpooling2D_output_shape(input_shape):
        shape = list(input_shape)
        shape[1] *= 2
        shape[2] *= 2
        return tuple(shape)
    
    
    Output_Tensor = Lambda(MaxUnpooling2D_func,MaxUnpooling2D_output_shape)(Input_Tensor)
    return Output_Tensor



def MaxUnpooling3D(Input_Tensor, Kernel_size=(2,2,2)):
    def MaxUnpooling3D_func(InputTensor):
        #in: b,o,n,m,c
        #out: b,2o,2n,2m,c
        #(batch_size, dim1, dim2, dim3, channels)
        batchsize=K.shape(InputTensor)[0]
        inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2], K.shape(InputTensor)[3])
        chan=K.shape(InputTensor)[4] #nchannels
        
        out=Input_Tensor
        
        if Kernel_size[2]==2:
            #Add zeros to contracted dim
            padded=K.concatenate((out,K.zeros_like(out)), axis=4) #b,o,n,m,2*c
            #Reshape so that ncols is spaced with zeros
            out = K.reshape(padded, (batchsize,inshape[0],inshape[1],inshape[2]*Kernel_size[2],chan)) #b,o,n,2*m,c
        
        if Kernel_size[1]==2:
            #Space nrows the same way
            padded=K.concatenate((out,K.zeros_like(out)), axis=3) # b,n,2*2*m,c
            out=K.reshape(padded, (batchsize,inshape[0],inshape[1]*Kernel_size[1],inshape[2]*Kernel_size[2],chan)) # b,o,2*n,2*m,c
        
        if Kernel_size[0]==2:
            #Space nrows the same way
            padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,o,2*2*n,2*m,c
            out=K.reshape(padded, (batchsize,inshape[0]*Kernel_size[0],inshape[1]*Kernel_size[1],inshape[2]*Kernel_size[2],chan)) # b,2*o,2*n,2*m,c
        
        return out

    def MaxUnpooling3D_output_shape(input_shape):
        shape = list(input_shape)
        shape[1] *= Kernel_size[0]
        shape[2] *= Kernel_size[1]
        shape[3] *= Kernel_size[2]
        return tuple(shape)
    
    Output_Tensor = Lambda(MaxUnpooling3D_func,MaxUnpooling3D_output_shape)(Input_Tensor)
    return Output_Tensor


#Legacy Code:

"""
def MaxUnpooling2Dalt(InputTensor):
    #in: 2,2 out: 4,4
    #[1,2],[3,4]
    
    #(2,2)
    #length=shape[0] #4
    resh = K.reshape(InputTensor, (4,1 )) #[1,2,3,4]

    padding = K.zeros((4,1))
    padded=K.concatenate((resh,padding), axis=1)#[1,0],[2,0],...
    
    out = K.reshape(padded, (2,4 ))#[1 0 1 0],[1,0,1,0]
    
    empty=K.zeros((2,4))
    padded2=K.concatenate((out,empty), axis=1) #101010000,0...,
    out2=K.reshape(padded2,(4,4,1))
    return out2

def MaxUnpooling2DFixedDim(InputTensor):
    #in: batch,2,2,1
    #out: batch,4,4,1
    #(batch_size, rows, cols, channels)

    #In form b,4,1
    resh = K.reshape(InputTensor, (K.shape(InputTensor)[0],)+(4,1)) #[1,2,3,4]
    
    padding = K.zeros_like(resh)
    padded=K.concatenate((resh,padding), axis=2)#[1,0],[2,0],...
    
    out = K.reshape(padded, (K.shape(padded)[0],)+(2,4 ))#[1 0 1 0],[1,0,1,0]
    
    empty=K.zeros_like(out)
    padded2=K.concatenate((out,empty), axis=2) #101010000,0...,
    out2=K.reshape(padded2, (K.shape(padded2)[0],)+(4,4,1))
    return out2

def MaxUnpooling2DChanOne(InputTensor):
    #in: batch,n,m,1
    #out: batch,4,6,1
    #(batch_size, rows, cols, channels)
    inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2]) #(2,3)
    
    #In form b,6,1
    resh = K.reshape(InputTensor, (K.shape(InputTensor)[0],)+(inshape[0]*inshape[1],1)) #[1,2,3,4]
    
    padding = K.zeros_like(resh)
    padded=K.concatenate((resh,padding), axis=2)#[1,0],[2,0],... shape b,6,2
    
    out = K.reshape(padded, (K.shape(padded)[0],)+(inshape[0],inshape[1]*2))#[1 0 1 0],[1,0,1,0]
    
    empty=K.zeros_like(out)
    padded2=K.concatenate((out,empty), axis=2) #101010000,0...,
    out2=K.reshape(padded2, (K.shape(padded2)[0],)+(inshape[0]*2,inshape[1]*2,1))
    return out2

def MaxUnpooling2Do(InputTensor):
    #in: b,n,m,c
    #out: b,2n,2m,c
    #(batch_size, rows, cols, channels)
    batchsize=K.shape(InputTensor)[0]
    inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2]) #(nrows,ncols)
    chan=K.shape(InputTensor)[3] #nchannels
    
    #Contract rows and cols
    resh = K.reshape(InputTensor, (batchsize,inshape[0]*inshape[1],chan)) #b,n*m,c
    
    #Add zeros to contracted dim
    padded=K.concatenate((resh,K.zeros_like(resh)), axis=2) #b,2*n*m,c
    #Reshape so that ncols is spaced with zeros
    out = K.reshape(padded, (batchsize, inshape[0],inshape[1]*2,chan)) #b,n,2*m,c
    
    #Space nrows the same way
    padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,n,2*2*m,c
    out=K.reshape(padded, (batchsize, inshape[0]*2,inshape[1]*2,chan)) # b,2*n,2*m,c
    
    return out

def MaxUnpooling2D(InputTensor):
    #in: b,n,m,c
    #out: b,2n,2m,c
    #(batch_size, rows, cols, channels)
    batchsize=K.shape(InputTensor)[0]
    inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2]) #(nrows,ncols)
    chan=K.shape(InputTensor)[3] #nchannels
    
    #Add zeros to dimension next to ncols that will be spaced (which is channels here) 
    padded=K.concatenate((InputTensor,K.zeros_like(InputTensor)), axis=3) #b,n,m,2*c
    #Reshape so that ncols is spaced with zeros
    out = K.reshape(padded, (batchsize, inshape[0],inshape[1]*2,chan)) #b,n,2*m,c
    
    #Space nrows the same way
    padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,n,2*2*m,c
    out=K.reshape(padded, (batchsize, inshape[0]*2,inshape[1]*2,chan)) # b,2*n,2*m,c
    
    return out

def MaxUnpooling2D_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    shape[2] *= 2
    return tuple(shape)

def MaxUnpooling3Do(InputTensor):
    #in: b,o,n,m,c
    #out: b,2o,2n,2m,c
    #(batch_size, dim1, dim2, dim3, channels)
    batchsize=K.shape(InputTensor)[0]
    inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2], K.shape(InputTensor)[3])
    chan=K.shape(InputTensor)[4] #nchannels
    
    #Contract rows and cols
    resh = K.reshape(InputTensor, (batchsize,inshape[0],inshape[1]*inshape[2],chan)) #b,o,n*m,c
    
    #Add zeros to contracted dim
    padded=K.concatenate((resh,K.zeros_like(resh)), axis=3) #b,o,2*n*m,c
    #Reshape so that ncols is spaced with zeros
    out = K.reshape(padded, (batchsize,inshape[0],inshape[1],inshape[2]*2,chan)) #b,o,n,2*m,c
    
    #Space nrows the same way
    padded=K.concatenate((out,K.zeros_like(out)), axis=3) # b,n,2*2*m,c
    out=K.reshape(padded, (batchsize,inshape[0],inshape[1]*2,inshape[2]*2,chan)) # b,o,2*n,2*m,c
    
    #Space nrows the same way
    padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,o,2*2*n,2*m,c
    out=K.reshape(padded, (batchsize,inshape[0]*2,inshape[1]*2,inshape[2]*2,chan)) # b,2*o,2*n,2*m,c
    
    return out




def MaxUnpooling3D(InputTensor):
    #in: b,o,n,m,c
    #out: b,2o,2n,2m,c
    #(batch_size, dim1, dim2, dim3, channels)
    batchsize=K.shape(InputTensor)[0]
    inshape=(K.shape(InputTensor)[1],K.shape(InputTensor)[2], K.shape(InputTensor)[3])
    chan=K.shape(InputTensor)[4] #nchannels

    #Add zeros to contracted dim
    padded=K.concatenate((InputTensor,K.zeros_like(InputTensor)), axis=4) #b,o,n,m,2*c
    #Reshape so that ncols is spaced with zeros
    out = K.reshape(padded, (batchsize,inshape[0],inshape[1],inshape[2]*2,chan)) #b,o,n,2*m,c
    
    #Space nrows the same way
    padded=K.concatenate((out,K.zeros_like(out)), axis=3) # b,n,2*2*m,c
    out=K.reshape(padded, (batchsize,inshape[0],inshape[1]*2,inshape[2]*2,chan)) # b,o,2*n,2*m,c
    
    #Space nrows the same way
    padded=K.concatenate((out,K.zeros_like(out)), axis=2) # b,o,2*2*n,2*m,c
    out=K.reshape(padded, (batchsize,inshape[0]*2,inshape[1]*2,inshape[2]*2,chan)) # b,2*o,2*n,2*m,c
    
    
    return out

def MaxUnpooling3D_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    shape[2] *= 2
    shape[3] *= 2
    return tuple(shape)
"""



