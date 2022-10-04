from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numpy import genfromtxt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.models import load_model
from tensorflow import keras


from skimage.metrics import structural_similarity as ssim
import seaborn as sns
import pandas as pd
import os

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def mat_hibrida(A,B):
    A_up=upper_tri_masking_mat(A)
    B_low=lower_tri_masking_mat(B)
    return A_up+B_low
    
def upper_tri_masking_mat(A):
    mat_mit = np.zeros((90,90))
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    mat_mit[mask] = A[mask]
    return mat_mit

def lower_tri_masking_mat(A):
    mat_mit = np.zeros((90,90))
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] > r
    mat_mit[mask] = A[mask]
    return mat_mit    


def model_to_output(models,
                 data, 
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    #filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    
    out = decoder.predict(z_mean)
    
    
    return out

# load data simulated
my_data = genfromtxt('fc_W_N3_UWS.csv',delimiter=',')
label = genfromtxt('label_W_N3_UWS.csv',delimiter=',')
x_train = my_data[0:int(len(my_data)*0.7)]
x_test= my_data[int(len(my_data)*0.7)+1:len(my_data)]
label_1 = label
y_train = label_1[0:int(len(my_data)*0.7)]
y_test = label_1[int(len(my_data)*0.7)+1:len(my_data)]

N1 = genfromtxt('FC_N1_multi.csv',delimiter=',')
N2 = genfromtxt('FC_N2_multi.csv',delimiter=',')
Inc = genfromtxt('FC_Inc.csv',delimiter=',')
sed = genfromtxt('FC_sed.csv',delimiter=',')
MCS = genfromtxt('FC_MCS.csv',delimiter=',')
UWS = genfromtxt('FC_UWS.csv',delimiter=',')
#Wp = genfromtxt('FC_Wprop.csv',delimiter=',')
#Wc = genfromtxt('FC_Wcoma.csv',delimiter=',')
N3 = x_test[np.where(y_test==1),:]
W =x_test[np.where(y_test==0),:]

N3=np.squeeze(N3[0,500:801,:])
W = np.squeeze(W[0,1:301,:])

FCemp = genfromtxt('FC_emp_full.csv',delimiter=',')

#-------------matrices hibridas EMP/MODEL
fig, ax = plt.subplots(ncols=4, nrows=2,figsize=(20,10))
plt.subplot(2,4,1)
# plotea
#ax.grid(False)
plt.imshow(mat_hibrida(W[10,:].reshape(90,90),FCemp[0,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
plt.clim(-0.2, 1)
cbar.ax.tick_params(labelsize=15)

plt.subplot(2,4,2)
#ax.grid(False)
plt.imshow(mat_hibrida(N1[10,:].reshape(90,90),FCemp[1,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.2, 1)

plt.subplot(2,4,3)
#ax.grid(False)
plt.imshow(mat_hibrida(N2[14,:].reshape(90,90),FCemp[2,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.2, 1)

plt.subplot(2,4,4)
#ax.grid(False)
plt.imshow(mat_hibrida(N3[10,:].reshape(90,90),FCemp[3,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.2, 1)

plt.subplot(2,4,5)
#ax.grid(False)
plt.imshow(mat_hibrida(sed[10,:].reshape(90,90),FCemp[4,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)

plt.subplot(2,4,6)
#ax.grid(False)
plt.imshow(mat_hibrida(Inc[10,:].reshape(90,90),FCemp[5,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)

plt.subplot(2,4,7)
#ax.grid(False)
plt.imshow(mat_hibrida(MCS[10,:].reshape(90,90),FCemp[6,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)

plt.subplot(2,4,8)
#ax.grid(False)
plt.imshow(mat_hibrida(UWS[10,:].reshape(90,90),FCemp[7,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)
plt.savefig('matrices_hibridas_emp_mod.svg',dpi=300)

#----------SSIM EMP/MODEL

sW=[]
sN1=[]
sN2=[]
sN3=[]
sSed=[]
sInc=[]
sMCS=[]
sUWS=[]

for i in range(len(W)):
    sim1=ssim(W[i,:].reshape(90,90),FCemp[0,:].reshape(90,90))
    sim2=ssim(N1[i,:].reshape(90,90),FCemp[1,:].reshape(90,90))
    sim3=ssim(N2[i,:].reshape(90,90),FCemp[2,:].reshape(90,90))
    sim4=ssim(N3[i,:].reshape(90,90),FCemp[3,:].reshape(90,90))
    sim5=ssim(sed[i,:].reshape(90,90),FCemp[4,:].reshape(90,90))
    sim6=ssim(Inc[i,:].reshape(90,90),FCemp[5,:].reshape(90,90))
    
    sim7=ssim(MCS[i,:].reshape(90,90),FCemp[6,:].reshape(90,90))
    sim8=ssim(UWS[i,:].reshape(90,90),FCemp[7,:].reshape(90,90))
    sW.append(sim1)
    sN1.append(sim2)
    sN2.append(sim3)
    sN3.append(sim4)
    sSed.append(sim5)
    sInc.append(sim6)
    sMCS.append(sim7)
    sUWS.append(sim8)

sW = np.array(sW)
sN1 = np.array(sN1)
sN2 = np.array(sN2)
sN3 = np.array(sN3)
sSed = np.array(sSed)
sInc = np.array(sInc)
sMCS = np.array(sMCS)
sUWS = np.array(sUWS)

data = [['W', sW] , ['N1', sN1], ['N2', sN2],['N3',sN3],['S',sSed],['LoC',sInc],['MCS',sMCS],['UWS',sUWS]]
df1 = pd.DataFrame(data, columns = ['State', 'SSIM'])

df1 = df1.explode('SSIM')
df1['SSIM'] = df1['SSIM'].astype('float')


sns.set_theme()
sns.set_context("talk")
plt.figure(figsize=(20,10))
sns.violinplot(x='State', y='SSIM', data=df1)
plt.ylim(0,1)

plt.savefig('violin_model_emp.svg',dpi=300)

#----Encodear las matrices modeladas y ver qué salen---#
# define el modelo

original_dim = 8100


# network parameters
input_shape = (original_dim, )
intermediate_dim = 1028
batch_size = 128
latent_dim = 2
epochs = 20

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')



models = (encoder, decoder)
data = (x_test, y_test)
reconstruction_loss = binary_crossentropy(inputs,outputs)
                                                
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# carga los pesos ya entrenados
vae.load_weights('vae_Liege.h5')



label_W = np.ones(300)*0
W_data = (W,label_W)
salida1 = model_to_output(models,
    W_data,
    batch_size=batch_size,
    model_name="vae_mlp")

label_W = np.ones(300)*0
N1_data = (N1,label_W)
salida2 = model_to_output(models,
    N1_data,
    batch_size=batch_size,
    model_name="vae_mlp")

label_W = np.ones(300)*0
N2_data = (N2,label_W)
salida3 = model_to_output(models,
    N2_data,
    batch_size=batch_size,
    model_name="vae_mlp")

label_W = np.ones(300)*0
N3_data = (N3,label_W)
salida4 = model_to_output(models,
    N3_data,
    batch_size=batch_size,
    model_name="vae_mlp")

label_W = np.ones(300)*0
Sed_data = (sed,label_W)
salida5 = model_to_output(models,
    Sed_data,
    batch_size=batch_size,
    model_name="vae_mlp")

label_W = np.ones(300)*0
Inc_data = (Inc,label_W)
salida6 = model_to_output(models,
    Inc_data,
    batch_size=batch_size,
    model_name="vae_mlp")

label_W = np.ones(300)*0
MCS_data = (MCS,label_W)
salida7 = model_to_output(models,
    MCS_data,
    batch_size=batch_size,
    model_name="vae_mlp")

label_W = np.ones(300)*0
UWS_data = (UWS,label_W)
salida8 = model_to_output(models,
    UWS_data,
    batch_size=batch_size,
    model_name="vae_mlp")



fig, ax = plt.subplots(ncols=4, nrows=2,figsize=(20,10))
plt.subplot(2,4,1)
# plotea
#ax.grid(False)
plt.imshow(mat_hibrida(W[10,:].reshape(90,90),salida1[0,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
plt.clim(-0.2, 1)
cbar.ax.tick_params(labelsize=15)

plt.subplot(2,4,2)
#ax.grid(False)
plt.imshow(mat_hibrida(N1[10,:].reshape(90,90),salida2[1,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.2, 1)

plt.subplot(2,4,3)
#ax.grid(False)
plt.imshow(mat_hibrida(N2[14,:].reshape(90,90),salida3[2,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.2, 1)

plt.subplot(2,4,4)
#ax.grid(False)
plt.imshow(mat_hibrida(N3[7,:].reshape(90,90),salida4[1,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.2, 1)

plt.subplot(2,4,5)
#ax.grid(False)
plt.imshow(mat_hibrida(sed[10,:].reshape(90,90),salida5[4,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)

plt.subplot(2,4,6)
#ax.grid(False)
plt.imshow(mat_hibrida(Inc[10,:].reshape(90,90),salida6[5,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)

plt.subplot(2,4,7)
#ax.grid(False)
plt.imshow(mat_hibrida(MCS[10,:].reshape(90,90),salida7[6,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)

plt.subplot(2,4,8)
#ax.grid(False)
plt.imshow(mat_hibrida(UWS[10,:].reshape(90,90),salida8[7,:].reshape(90,90)))
cbar = plt.colorbar(shrink=0.5)
cbar.ax.tick_params(labelsize=15)
plt.clim(-0.4, 1)

plt.savefig('matrices_mod_vae.svg',dpi=300)

sW=[]
sN1=[]
sN2=[]
sN3=[]
sSed=[]
sInc=[]
sMCS=[]
sUWS=[]

for i in range(len(W)):
    sim1=ssim(W[i,:].reshape(90,90),salida1[i,:].reshape(90,90))
    sim2=ssim(N1[i,:].reshape(90,90),salida2[i,:].reshape(90,90))
    sim3=ssim(N2[i,:].reshape(90,90),salida3[i,:].reshape(90,90))
    sim4=ssim(N3[i,:].reshape(90,90),salida4[i,:].reshape(90,90))
    sim5=ssim(sed[i,:].reshape(90,90),salida5[i,:].reshape(90,90))
    sim6=ssim(Inc[i,:].reshape(90,90),salida6[i,:].reshape(90,90))
    
    sim7=ssim(MCS[i,:].reshape(90,90),salida7[i,:].reshape(90,90))
    sim8=ssim(UWS[i,:].reshape(90,90),salida8[i,:].reshape(90,90))
    sW.append(sim1)
    sN1.append(sim2)
    sN2.append(sim3)
    sN3.append(sim4)
    sSed.append(sim5)
    sInc.append(sim6)
    sMCS.append(sim7)
    sUWS.append(sim8)

sWvae = np.array(sW)
sN1vae = np.array(sN1)
sN2vae = np.array(sN2)
sN3vae = np.array(sN3)
sSedvae = np.array(sSed)
sIncvae = np.array(sInc)
sMCSvae = np.array(sMCS)
sUWSvae = np.array(sUWS)


data = [['W', sWvae] , ['N1', sN1vae], ['N2', sN2vae],['N3',sN3vae],['S',sSedvae],['LoC',sIncvae],['MCS',sMCSvae],['UWS',sUWSvae]]
df2 = pd.DataFrame(data, columns = ['State', 'SSIM'])

df2 = df2.explode('SSIM')
df2['SSIM'] = df2['SSIM'].astype('float')

sns.set_theme()
sns.set_context("talk")
plt.figure(figsize=(20,10))
sns.violinplot(x='State', y='SSIM', data=df2)
plt.ylim(0,1)
plt.savefig('violin_mod_vae.svg',dpi=300)