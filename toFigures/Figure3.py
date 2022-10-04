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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

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

def plot_data_in_latent(models,
                 data, color,nombre,
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

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    #plt.figure(figsize=(10, 10))
#    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=color,alpha = 0.8,label=nombre)
    #plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.savefig(filename)
    plt.show()
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.legend()
    return z_mean

# MNIST datase
def plot_pert_in_latent(models,
                 data,color,symbol,plotsi,
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

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
   # plt.figure(figsize=(12, 10))
#    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    if plotsi==1:
        plt.scatter(z_mean[:, 0], z_mean[:, 1],marker=symbol, c=data[1], alpha = 0.3,cmap='viridis')
        #plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.savefig(filename)
        plt.show()
        plt.xlim(-4,4)
        plt.ylim(-4,4)
    #print(z_mean)
    return z_mean
def drawArrow(A, B):

    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=0.1, length_includes_head=True,color='b')

def module_arrow(A, B):
    return np.sqrt((B[0] - A[0])**2+ (B[1] - A[1])**2)

W = genfromtxt('W_multi.txt')
N1 = genfromtxt('FC_N1_multi.csv',delimiter=',')
N2 = genfromtxt('FC_N2_multi.csv',delimiter=',')
Inc = genfromtxt('FC_Inc.csv',delimiter=',')
sed = genfromtxt('FC_sed.csv',delimiter=',')
MCS = genfromtxt('FC_MCS.csv',delimiter=',')
UWS = genfromtxt('FC_UWS.csv',delimiter=',')
N31 = genfromtxt('FC_N3_multi.csv',delimiter=',')
#W1 = genfromtxt('FC_W_multi.csv',delimiter=',')



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



# carga los pesos ya entrenados
vae.load_weights('vae_Liege.h5')


# hace la figura con todos los puntos encodeados



plt.figure(figsize=(10,10))
label_W = np.ones(300)*0
W_data = (W,label_W)
zz = plot_data_in_latent(models,
    W_data,'b','W',
    batch_size=batch_size,
    model_name="vae_mlp")
W_centroid = np.mean(zz,0)

label_N1 = np.ones(300)*2
N1_data = (N1,label_N1)
zz=plot_data_in_latent(models,
    N1_data,'purple','N1',
    batch_size=batch_size,
    model_name="vae_mlp")
N1_centroid = np.mean(zz,0)

label_N2= np.ones(300)*2.5
N2_data = (N2,label_N2)
zz=plot_data_in_latent(models,
    N2_data,'yellow','N2',
    batch_size=batch_size,
    model_name="vae_mlp")
N2_centroid = np.mean(zz,0)

N31 = np.squeeze(N31)
label_N31= np.ones(300)*1
N31_data = (N31,label_N31)
zz=plot_data_in_latent(models,
    N31_data,'orange','N3',
    batch_size=batch_size,
    model_name="vae_mlp")
N3_centroid = np.mean(zz,0)


label_sed = np.ones(300)*4
sed_data = (sed,label_sed)
zz  =plot_data_in_latent(models,
    sed_data,'g','Sed',
    batch_size=batch_size,
    model_name="vae_mlp")
sed_centroid = np.mean(zz,0)

label_Inc = np.ones(300)*3
Inc_data = (Inc,label_Inc)
zz= plot_data_in_latent(models,
    Inc_data,'r','LoC',
    batch_size=batch_size,
    model_name="vae_mlp")
Inc_centroid = np.mean(zz,0)


label_MCS = np.ones(300)*5
MCS_data = (MCS,label_MCS)
zz = plot_data_in_latent(models,
    MCS_data,'tan','MCS',
    batch_size=batch_size,
    model_name="vae_mlp")
MCS_centroid = np.mean(zz,0)

label_UWS = np.ones(300)*6
UWS_data = (UWS,label_UWS)
zz = plot_data_in_latent(models,
    UWS_data,'c','UWS',
    batch_size=batch_size,
    model_name="vae_mlp")
UWS_centroid = np.mean(zz,0)

# label_Wp = np.ones(300)*6
# Wp_data = (Wp,label_Wp)
# zz = plot_data_in_latent(models,
#     Wp_data,'mediumpurple','Wp',
#     batch_size=batch_size,
#     model_name="vae_mlp")
# Wp_centroid = np.mean(zz,0)


# label_Wc = np.ones(300)*6
# Wc_data = (Wc,label_Wc)
# zz = plot_data_in_latent(models,
#     Wc_data,'k','Wc',
#     batch_size=batch_size,
#     model_name="vae_mlp")
# Wc_centroid = np.mean(zz,0)

x_cent = [UWS_centroid[0], MCS_centroid[0],Inc_centroid[0],sed_centroid[0],N3_centroid[0],N2_centroid[0],N1_centroid[0],W_centroid[0]]
y_cent = [UWS_centroid[1], MCS_centroid[1],Inc_centroid[1],sed_centroid[1],N3_centroid[1],N2_centroid[1],N1_centroid[1],W_centroid[1]]


xvals = np.linspace(UWS_centroid[0], W_centroid[0], 100)
yinterp = np.interp(xvals, x_cent, y_cent)

plt.plot(xvals,yinterp)
plt.scatter(x_cent,y_cent)


# importa todas las perturbaciones: esto ya sería fig 3






import scipy.io

mat = scipy.io.loadmat('kick_N1_full.mat')
kick_full_N1=mat['FC_sim_kick']

mat = scipy.io.loadmat('kick_N2_full.mat')
kick_full_N2=mat['FC_sim_kick']

mat = scipy.io.loadmat('kick_N3_full.mat')
kick_full_N3=mat['FC_sim_kick']

mat = scipy.io.loadmat('kick_Inc_full.mat')
kick_full_Inc=mat['FC_sim_kick']

mat = scipy.io.loadmat('kick_sed_full.mat')
kick_full_sed=mat['FC_sim_kick']

mat = scipy.io.loadmat('FC_kick_full_MCS.mat')
kick_full_MCS=mat['FC_sim_kick']

mat = scipy.io.loadmat('kick_MCS_full2.mat')
kick_full_MCS=mat['FC_sim_kick']

mat = scipy.io.loadmat('FC_kick_full_UWS.mat')
kick_full_UWS=mat['FC_sim_kick']

mat = scipy.io.loadmat('kick_UWS_full.mat')
kick_full_UWS=mat['FC_sim_kick']



label_kick = np.arange(0,20)
kick_N1 = np.squeeze(kick_full_N1.reshape(1,8100,20,45))
zz =[]
for kk in range(1):
    kick = np.transpose(kick_N1[:,:,kk+3])
    kick_data = (kick,label_kick)
    zz1 = plot_pert_in_latent(models,
             kick_data,'y','^',1,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_N1 = np.asarray(zz)
pert_N1_centroid = np.mean(pert_N1,0)
plt.scatter(N1_centroid[0],N1_centroid[1])

kick_MCS = np.squeeze(kick_full_MCS.reshape(1,8100,20,45))
zz = []
for kk in range(45):
    kick = np.transpose(kick_MCS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
            kick_data,'m','x',0,
            batch_size=batch_size,
            model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_MCS = np.asarray(zz)
pert_MCS_centroid = np.mean(pert_MCS,0)

kick_UWS = np.squeeze(kick_full_UWS.reshape(1,8100,20,45))
zz = []
for kk in range(45):
    kick = np.transpose(kick_UWS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'r','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_UWS= np.asarray(zz)
pert_UWS_centroid = np.mean(pert_UWS,0)


kick_N2 = np.squeeze(kick_full_N2.reshape(1,8100,20,45))
zz =[]
for kk in range(45):
    kick = np.transpose(kick_N2[:,:,kk])
    kick_data = (kick,label_kick)
    zz1 = plot_pert_in_latent(models,
             kick_data,'r','*',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_N2 = np.asarray(zz)
pert_N2_centroid = np.mean(pert_N2,0)


kick_N3 = np.squeeze(kick_full_N3.reshape(1,8100,20,45))
zz =[]
for kk in range(45):
    kick = np.transpose(kick_N3[:,:,kk])
    kick_data = (kick,label_kick)
    zz1 = plot_pert_in_latent(models,
             kick_data,'y','^',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_N3 = np.asarray(zz)
pert_N3_centroid = np.mean(pert_N3,0)

kick_Inc = np.squeeze(kick_full_Inc.reshape(1,8100,20,45))
# encode the kicks for Inc
zz = []
for kk in range(45):
    kick = np.transpose(kick_Inc[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'r','^',1,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_inc = np.asarray(zz)
pert_inc_centroid = np.mean(pert_inc,0)

kick_sed = np.squeeze(kick_full_sed.reshape(1,8100,20,45))
zz = []
for kk in range(45):
    kick = np.transpose(kick_sed[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_sed= np.asarray(zz)
pert_sed_centroid = np.mean(pert_sed,0)

## FIGURA 4-----------------------
# ----------grafica solo las flechas---------
sns.set_theme()
sns.set()
sns.set(font_scale = 1.5)

plt.figure(figsize=(10, 10))
drawArrow(Inc_centroid, pert_inc_centroid)   
drawArrow(UWS_centroid, pert_UWS_centroid) 
drawArrow(sed_centroid, pert_sed_centroid) 
drawArrow(N3_centroid, pert_N3_centroid)
drawArrow(N2_centroid, pert_N2_centroid)
drawArrow(N1_centroid, pert_N1_centroid)
drawArrow(MCS_centroid, pert_MCS_centroid)
plt.xlim(-4,4)
plt.ylim(-4,4)

xvals_sub = np.linspace(UWS_centroid[0], W_centroid[0], 10)
yinterp_sub = np.interp(xvals_sub, x_cent, y_cent)

plt.plot(xvals,yinterp,linewidth=3,color='orange', linestyle='dashed')
plt.scatter(x_cent,y_cent)
plt.scatter(xvals_sub,yinterp_sub,s=150,marker='*',color='r')

plt.savefig('arrow_latent44.svg',dpi=300)

plt.savefig('scheme_D2M.svg',dpi=300)


# ---------------compute arrow module
ModArrow=np.zeros(7)
ModArrow[4]=module_arrow(Inc_centroid, pert_inc_centroid)   
ModArrow[6]=module_arrow(UWS_centroid, pert_UWS_centroid) 
ModArrow[3]=module_arrow(sed_centroid, pert_sed_centroid) 
ModArrow[2]=module_arrow(N3_centroid, pert_N3_centroid)
ModArrow[1]=module_arrow(N2_centroid, pert_N2_centroid)
ModArrow[0]=module_arrow(N1_centroid, pert_N1_centroid)
ModArrow[5]=module_arrow(MCS_centroid, pert_MCS_centroid)


# ---------------compute distance to manifold and to W


def dist_to_manifold(xvals,yinterp,perturb):
    dist_cum = []   
    for ii in range(len(xvals)):
        dist=np.sqrt((xvals[ii]-perturb[0])**2+(yinterp[ii]-perturb[1])**2)
        dist_cum.append(dist)
    min_dist = dist_cum[np.argmin(dist_cum)] 
    return min_dist

# perturbation average
d2M=np.zeros(7);
d2M[0]= dist_to_manifold(xvals,yinterp,pert_N1_centroid)
d2M[1]=dist_to_manifold(xvals,yinterp,pert_N2_centroid)
d2M[2]=dist_to_manifold(xvals,yinterp,pert_N3_centroid)
d2M[3]=dist_to_manifold(xvals,yinterp,pert_sed_centroid)
d2M[4]=dist_to_manifold(xvals,yinterp,pert_inc_centroid)
d2M[5]=dist_to_manifold(xvals,yinterp,pert_MCS_centroid)
d2M[6]=dist_to_manifold(xvals,yinterp,pert_UWS_centroid)


d2W=np.zeros(7)
d2W[0]=  np.sqrt((N1_centroid[0]-W_centroid[0])**2+(N1_centroid[1]-W_centroid[1])**2)
d2W[1]=  np.sqrt((N2_centroid[0]-W_centroid[0])**2+(N2_centroid[1]-W_centroid[1])**2)
d2W[2]=  np.sqrt((N3_centroid[0]-W_centroid[0])**2+(N3_centroid[1]-W_centroid[1])**2)
d2W[3]= np.sqrt((sed_centroid[0]-W_centroid[0])**2+(sed_centroid[1]-W_centroid[1])**2)
d2W[4]=  np.sqrt((Inc_centroid[0]-W_centroid[0])**2+(Inc_centroid[1]-W_centroid[1])**2)
d2W[5]=  np.sqrt((MCS_centroid[0]-W_centroid[0])**2+(MCS_centroid[1]-W_centroid[1])**2)
d2W[6]=  np.sqrt((UWS_centroid[0]-W_centroid[0])**2+(UWS_centroid[1]-W_centroid[1])**2)

dP2W=np.zeros(7)
dP2W[0]=  np.sqrt((pert_N1_centroid[0]-W_centroid[0])**2+(pert_N1_centroid[1]-W_centroid[1])**2)
dP2W[1]=  np.sqrt((pert_N2_centroid[0]-W_centroid[0])**2+(pert_N2_centroid[1]-W_centroid[1])**2)
dP2W[2]=  np.sqrt((pert_N3_centroid[0]-W_centroid[0])**2+(pert_N3_centroid[1]-W_centroid[1])**2)
dP2W[3]= np.sqrt((pert_sed_centroid[0]-W_centroid[0])**2+(pert_sed_centroid[1]-W_centroid[1])**2)
dP2W[4]=  np.sqrt((pert_inc_centroid[0]-W_centroid[0])**2+(pert_inc_centroid[1]-W_centroid[1])**2)
dP2W[5]=  np.sqrt((pert_MCS_centroid[0]-W_centroid[0])**2+(pert_MCS_centroid[1]-W_centroid[1])**2)
dP2W[6]=  np.sqrt((pert_UWS_centroid[0]-W_centroid[0])**2+(pert_UWS_centroid[1]-W_centroid[1])**2)


plt.figure(figsize=(10, 10))
plt.plot(dP2W,marker='o',
     markerfacecolor='blue',color='b', markersize=12)
plt.savefig('d2Wav.svg',dpi=300)


plt.figure(figsize=(10, 10))
plt.plot(d2W,marker='o',
     markerfacecolor='green',color='g', markersize=12)
plt.savefig('d2Worav.svg',dpi=300)


plt.figure(figsize=(10, 10))
plt.plot(d2M,marker='o',
     markerfacecolor='orange', color='orange',markersize=12)

plt.savefig('d2Mav.svg',dpi=300)

plt.figure(figsize=(10, 10))
plt.plot(ModArrow,marker='o',
     markerfacecolor='orange', color='orange',markersize=12)

plt.savefig('ModArrow.svg',dpi=300)


## --FIgura 5

# ----------node by node perturbed-------------------
dist_full_nodesM = np.zeros((7,45))  # distantnace to Manifold
dist_full_nodesW = np.zeros((7,45))  # distance to Wake
dist_full_nodesModule = np.zeros((7,45))  # distance to origin

dist_min_node2M=[]
for kk in range(45):
    kick = np.transpose(kick_N1[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = dist_to_manifold(xvals,yinterp,enco_pert)
    dist_min_node2M.append(distance)
    
dist_min_node2W=[]
dist_min_node2Ori=[]
enco_pert_full=[]
for kk in range(45):
    kick = np.transpose(kick_N1[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = np.sqrt((enco_pert[0]-W_centroid[0])**2+(enco_pert[1]-W_centroid[1])**2)
    dist2ori = module_arrow(enco_pert,N1_centroid)
    dist_min_node2W.append(distance)
    enco_pert_full.append(enco_pert)
    dist_min_node2Ori.append(dist2ori)
dist_matrix_N1 =pairwise_distances(enco_pert_full,metric='euclidean')    
dist_min_node2W = np.array(dist_min_node2W)
dist_min_node2Ori = np.array(dist_min_node2Ori)




dist_full_nodesM[0,:]=dist_min_node2M
dist_full_nodesW[0,:]=dist_min_node2W
dist_full_nodesModule[0,:]=dist_min_node2Ori

dist_min_node2M=[]
for kk in range(45):
    kick = np.transpose(kick_N2[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = dist_to_manifold(xvals,yinterp,enco_pert)
    dist_min_node2M.append(distance)
    
dist_min_node2W=[]
dist_min_node2Ori=[]
enco_pert_full=[]
for kk in range(45):
    kick = np.transpose(kick_N2[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = np.sqrt((enco_pert[0]-W_centroid[0])**2+(enco_pert[1]-W_centroid[1])**2)
    dist2ori = module_arrow(enco_pert,N2_centroid)
    dist_min_node2W.append(distance)
    enco_pert_full.append(enco_pert)
    dist_min_node2Ori.append(dist2ori)
dist_matrix_N2 =pairwise_distances(enco_pert_full,metric='euclidean')
dist_min_node2W = np.array(dist_min_node2W)
dist_min_node2Ori = np.array(dist_min_node2Ori)

dist_full_nodesM[1,:]=dist_min_node2M
dist_full_nodesW[1,:]=dist_min_node2W
dist_full_nodesModule[1,:]=dist_min_node2Ori

dist_min_node2M=[]
for kk in range(45):
    kick = np.transpose(kick_N3[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = dist_to_manifold(xvals,yinterp,enco_pert)
    dist_min_node2M.append(distance)
    
dist_min_node2W=[]
enco_pert_full=[]
dist_min_node2Ori=[]
for kk in range(45):
    kick = np.transpose(kick_N3[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = np.sqrt((enco_pert[0]-W_centroid[0])**2+(enco_pert[1]-W_centroid[1])**2)
    dist2ori = module_arrow(enco_pert,N3_centroid)
    dist_min_node2W.append(distance)
    enco_pert_full.append(enco_pert)
    dist_min_node2Ori.append(dist2ori)
dist_matrix_N3 =pairwise_distances(enco_pert_full,metric='euclidean')
dist_min_node2W = np.array(dist_min_node2W)
dist_min_node2Ori = np.array(dist_min_node2Ori)

dist_full_nodesM[2,:]=dist_min_node2M
dist_full_nodesW[2,:]=dist_min_node2W
dist_full_nodesModule[2,:]=dist_min_node2Ori

dist_min_node2M=[]
for kk in range(45):
    kick = np.transpose(kick_sed[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = dist_to_manifold(xvals,yinterp,enco_pert)
    dist_min_node2M.append(distance)
    
dist_min_node2W=[]
enco_pert_full=[]
dist_min_node2Ori=[]
for kk in range(45):
    kick = np.transpose(kick_sed[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = np.sqrt((enco_pert[0]-W_centroid[0])**2+(enco_pert[1]-W_centroid[1])**2)
    dist2ori = module_arrow(enco_pert,sed_centroid)
    dist_min_node2W.append(distance)
    enco_pert_full.append(enco_pert)
    dist_min_node2Ori.append(dist2ori)
dist_matrix_sed =pairwise_distances(enco_pert_full,metric='euclidean')
dist_min_node2W = np.array(dist_min_node2W)
dist_min_node2Ori = np.array(dist_min_node2Ori)

dist_full_nodesM[3,:]=dist_min_node2M
dist_full_nodesW[3,:]=dist_min_node2W
dist_full_nodesModule[3,:]=dist_min_node2Ori


dist_min_node2M=[]
for kk in range(45):
    kick = np.transpose(kick_Inc[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = dist_to_manifold(xvals,yinterp,enco_pert)
    dist_min_node2M.append(distance)
    
dist_min_node2W=[]
enco_pert_full=[]
dist_min_node2Ori=[]
for kk in range(45):
    kick = np.transpose(kick_Inc[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = np.sqrt((enco_pert[0]-W_centroid[0])**2+(enco_pert[1]-W_centroid[1])**2)
    dist_min_node2W.append(distance)
    enco_pert_full.append(enco_pert)
    dist2ori = module_arrow(enco_pert,Inc_centroid)
    dist_min_node2Ori.append(dist2ori)
dist_matrix_Inc =pairwise_distances(enco_pert_full,metric='euclidean')    
dist_min_node2W = np.array(dist_min_node2W)
dist_min_node2Ori = np.array(dist_min_node2Ori)

dist_full_nodesM[4,:]=dist_min_node2M
dist_full_nodesW[4,:]=dist_min_node2W
dist_full_nodesModule[4,:]=dist_min_node2Ori


dist_min_node2M=[]
for kk in range(45):
    kick = np.transpose(kick_MCS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = dist_to_manifold(xvals,yinterp,enco_pert)
    dist_min_node2M.append(distance)
    
dist_min_node2W=[]
enco_pert_full=[]
dist_min_node2Ori=[]
for kk in range(45):
    kick = np.transpose(kick_MCS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = np.sqrt((enco_pert[0]-W_centroid[0])**2+(enco_pert[1]-W_centroid[1])**2)
    dist_min_node2W.append(distance)
    enco_pert_full.append(enco_pert)
    dist2ori = module_arrow(enco_pert,MCS_centroid)
    dist_min_node2Ori.append(dist2ori)
dist_matrix_MCS =pairwise_distances(enco_pert_full,metric='euclidean')
dist_min_node2W = np.array(dist_min_node2W)
dist_min_node2Ori = np.array(dist_min_node2Ori)

dist_full_nodesM[5,:]=dist_min_node2M
dist_full_nodesW[5,:]=dist_min_node2W
dist_full_nodesModule[5,:]=dist_min_node2Ori

dist_min_node2M=[]
for kk in range(45):
    kick = np.transpose(kick_UWS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = dist_to_manifold(xvals,yinterp,enco_pert)
    dist_min_node2M.append(distance)
    
dist_min_node2W=[]
enco_pert_full=[]
dist_min_node2Ori=[]
for kk in range(45):
    kick = np.transpose(kick_UWS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    distance = np.sqrt((enco_pert[0]-W_centroid[0])**2+(enco_pert[1]-W_centroid[1])**2)
    dist_min_node2W.append(distance)
    enco_pert_full.append(enco_pert)
    dist2ori = module_arrow(enco_pert,MCS_centroid)
    dist_min_node2Ori.append(dist2ori)
dist_matrix_UWS =pairwise_distances(enco_pert_full,metric='euclidean')
dist_min_node2W = np.array(dist_min_node2W)
dist_min_node2Ori = np.array(dist_min_node2Ori)

dist_full_nodesM[6,:]=dist_min_node2M
dist_full_nodesW[6,:]=dist_min_node2W
dist_full_nodesModule[6,:]=dist_min_node2Ori

#np.savetxt('dist_full_nodes2M.txt',dist_full_nodesM)
#np.savetxt('dist_full_nodes2W.txt',dist_full_nodesW)
np.savetxt('dist_full_nodes2Ori.txt',dist_full_nodesModule)



np.savetxt('dist_pairwise_N1.txt',dist_matrix_N1)
np.savetxt('dist_pairwise_N2.txt',dist_matrix_N2)
np.savetxt('dist_pairwise_N3.txt',dist_matrix_N3)
np.savetxt('dist_pairwise_sed.txt',dist_matrix_sed)
np.savetxt('dist_pairwise_Inc.txt',dist_matrix_Inc)
np.savetxt('dist_pairwise_MCS.txt',dist_matrix_MCS)
np.savetxt('dist_pairwise_UWS.txt',dist_matrix_UWS)

## Computar en todo el trayecto la dist promedio 



label_kick = np.arange(0,20)
kick_N1 = np.squeeze(kick_full_N1.reshape(1,8100,20,45))
zz =[]
dist_full = []
dist_to_W = []
for kk in range(45):
    kick = np.transpose(kick_N1[:,:,kk])
    kick_data = (kick,label_kick)
    zz1 = plot_pert_in_latent(models,
             kick_data,'y','^',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    for ii in range(20):
        distance = np.sqrt((zz1[ii,0]-N1_centroid[0])**2+(zz1[ii,1]-N1_centroid[1])**2)
        dist_full.append(distance)
        distance2 = np.sqrt((zz1[ii,0]-W_centroid[0])**2+(zz1[ii,1]-W_centroid[1])**2)
        dist_to_W.append(distance2)        
dist_full = np.array(dist_full)
N1_dist_mean = np.mean(dist_full.reshape(45,20),0)
dist_to_W = np.array(dist_to_W)
N1_dist_to_W_mean = np.mean(dist_to_W.reshape(45,20),0)


kick_MCS = np.squeeze(kick_full_MCS.reshape(1,8100,20,45))
zz = []
dist_full = []
dist_to_W = []
for kk in range(45):
    kick = np.transpose(kick_MCS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
            kick_data,'m','x',0,
            batch_size=batch_size,
            model_name="vae_mlp")
    for ii in range(20):
        distance = np.sqrt((zz1[ii,0]-MCS_centroid[0])**2+(zz1[ii,1]-MCS_centroid[1])**2)
        dist_full.append(distance)
        distance2 = np.sqrt((zz1[ii,0]-W_centroid[0])**2+(zz1[ii,1]-W_centroid[1])**2)
        dist_to_W.append(distance2)         
dist_full = np.array(dist_full)
MCS_dist_mean = np.mean(dist_full.reshape(45,20),0)
dist_to_W = np.array(dist_to_W)
MCS_dist_to_W_mean = np.mean(dist_to_W.reshape(45,20),0)

kick_UWS = np.squeeze(kick_full_UWS.reshape(1,8100,20,45))
zz = []
dist_full = []
dist_to_W = []
for kk in range(45):
    kick = np.transpose(kick_UWS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'r','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    for ii in range(20):
        distance = np.sqrt((zz1[ii,0]-UWS_centroid[0])**2+(zz1[ii,1]-UWS_centroid[1])**2)
        dist_full.append(distance)
        distance2 = np.sqrt((zz1[ii,0]-W_centroid[0])**2+(zz1[ii,1]-W_centroid[1])**2)
        dist_to_W.append(distance2) 
dist_full = np.array(dist_full)
UWS_dist_mean = np.mean(dist_full.reshape(45,20),0)
dist_to_W = np.array(dist_to_W)
UWS_dist_to_W_mean = np.mean(dist_to_W.reshape(45,20),0)


kick_N2 = np.squeeze(kick_full_N2.reshape(1,8100,20,45))
zz =[]
dist_full = []
dist_to_W = []
for kk in range(45):
    kick = np.transpose(kick_N2[:,:,kk])
    kick_data = (kick,label_kick)
    zz1 = plot_pert_in_latent(models,
             kick_data,'r','*',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    for ii in range(20):
        distance = np.sqrt((zz1[ii,0]-N2_centroid[0])**2+(zz1[ii,1]-N2_centroid[1])**2)
        dist_full.append(distance)
        distance2 = np.sqrt((zz1[ii,0]-W_centroid[0])**2+(zz1[ii,1]-W_centroid[1])**2)
        dist_to_W.append(distance2) 
dist_full = np.array(dist_full)
N2_dist_mean = np.mean(dist_full.reshape(45,20),0)
dist_to_W = np.array(dist_to_W)
N2_dist_to_W_mean = np.mean(dist_to_W.reshape(45,20),0)


kick_N3 = np.squeeze(kick_full_N3.reshape(1,8100,20,45))
zz =[]
dist_full = []
dist_to_W = []
for kk in range(45):
    kick = np.transpose(kick_N3[:,:,kk])
    kick_data = (kick,label_kick)
    zz1 = plot_pert_in_latent(models,
             kick_data,'y','^',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    for ii in range(20):
        distance = np.sqrt((zz1[ii,0]-N3_centroid[0])**2+(zz1[ii,1]-N3_centroid[1])**2)
        dist_full.append(distance)
        distance2 = np.sqrt((zz1[ii,0]-W_centroid[0])**2+(zz1[ii,1]-W_centroid[1])**2)
        dist_to_W.append(distance2) 
dist_full = np.array(dist_full)
N3_dist_mean = np.mean(dist_full.reshape(45,20),0)
dist_to_W = np.array(dist_to_W)
N3_dist_to_W_mean = np.mean(dist_to_W.reshape(45,20),0)

kick_Inc = np.squeeze(kick_full_Inc.reshape(1,8100,20,45))
# encode the kicks for Inc
zz = []
dist_full = []
dist_to_W = []
for kk in range(45):
    kick = np.transpose(kick_Inc[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'r','^',1,
             batch_size=batch_size,
             model_name="vae_mlp")
    for ii in range(20):
        distance = np.sqrt((zz1[ii,0]-Inc_centroid[0])**2+(zz1[ii,1]-Inc_centroid[1])**2)
        dist_full.append(distance)
        distance2 = np.sqrt((zz1[ii,0]-W_centroid[0])**2+(zz1[ii,1]-W_centroid[1])**2)
        dist_to_W.append(distance2) 
dist_full = np.array(dist_full)
Inc_dist_mean = np.mean(dist_full.reshape(45,20),0)
dist_to_W = np.array(dist_to_W)
Inc_dist_to_W_mean = np.mean(dist_to_W.reshape(45,20),0)

kick_sed = np.squeeze(kick_full_sed.reshape(1,8100,20,45))
zz = []
dist_full = []
dist_to_W = []
for kk in range(45):
    kick = np.transpose(kick_sed[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
             kick_data,'g','x',0,
             batch_size=batch_size,
             model_name="vae_mlp")
    for ii in range(20):
        distance = np.sqrt((zz1[ii,0]-sed_centroid[0])**2+(zz1[ii,1]-sed_centroid[1])**2)
        dist_full.append(distance)
        distance2 = np.sqrt((zz1[ii,0]-W_centroid[0])**2+(zz1[ii,1]-W_centroid[1])**2)
        dist_to_W.append(distance2) 
dist_full = np.array(dist_full)
sed_dist_mean = np.mean(dist_full.reshape(45,20),0)
dist_to_W = np.array(dist_to_W)
sed_dist_to_W_mean = np.mean(dist_to_W.reshape(45,20),0)