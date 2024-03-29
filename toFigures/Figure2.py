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
def plot_results_out(models,
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

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the FC classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    colormap = np.array(['b', 'orange', 'c'])
    plt.figure(figsize=(10, 10))
    aver = np.array(y_test)
    aver = aver.astype(int)
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=colormap[aver],alpha=0.3)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    #plt.savefig('encode_test_set.svg',dpi=300)
    plt.show()

    filename = os.path.join(model_name, "FCs_over_latent.png")
    # display a 30x30 2D manifold of FCs
    n = 20
    digit_size = 90
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of FC classes in the latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    xdeco_cum=[]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
            xdeco_cum.append(x_decoded)
    xdeco_cum = np.array(xdeco_cum)
    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure,cmap='magma') #cmap='Greys_r')
    #plt.colorbar()
    plt.grid(color='w', linewidth=2)
    plt.clim(-0.1, 1)
    #plt.savefig('multiverse.svg',dpi=300)
    plt.show()
    return xdeco_cum

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
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.legend()
    return z_mean

# MNIST datase
def plot_pert_in_latent(models,
                 data,color,symbol,
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

    plt.scatter(z_mean[:, 0], z_mean[:, 1],marker=symbol, c=data[1], alpha = 0.3,cmap='viridis')
    #plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    return z_mean
def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=0.5, length_includes_head=True)

# load data simulated
my_data = genfromtxt('fc_W_N3_UWS.csv',delimiter=',')
label = genfromtxt('label_W_N3_UWS.csv',delimiter=',')
x_train = my_data[0:int(len(my_data)*0.7)]
x_test= my_data[int(len(my_data)*0.7)+1:len(my_data)]
label_1 = label
y_train = label_1[0:int(len(my_data)*0.7)]
y_test = label_1[int(len(my_data)*0.7)+1:len(my_data)]

W = genfromtxt('W_multi.txt')
N1 = genfromtxt('FC_N1_multi.csv',delimiter=',')
N2 = genfromtxt('FC_N2_multi.csv',delimiter=',')
Inc = genfromtxt('FC_Inc.csv',delimiter=',')
sed = genfromtxt('FC_sed.csv',delimiter=',')
MCS = genfromtxt('FC_MCS.csv',delimiter=',')
UWS = genfromtxt('FC_UWS.csv',delimiter=',')
N31 = genfromtxt('FC_N3_multi.csv',delimiter=',')
W1 = genfromtxt('FC_W_multi.csv',delimiter=',')
#Wp = genfromtxt('FC_Wprop.csv',delimiter=',')
#Wc = genfromtxt('FC_Wcoma.csv',delimiter=',')

# Esto es para usar W y N3 del corpus gigante que se uso para entrenar
# N3 = x_test[np.where(y_test==1),:]
W =x_test[np.where(y_test==0),:]

# N3=np.squeeze(N3[0,500:801,:])
W = np.squeeze(W[0,1:301,:])

FCemp = genfromtxt('FC_emp_full.csv',delimiter=',')

x_test=W
y_test=label_1[0:len(x_test)]

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



# plot results: the latent space with test set

data = (x_test, y_test)
zz = plot_results_out(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp")

zz = np.squeeze(zz)
np.savetxt('FCs_multi_20by20.txt',zz)

# hace la figura con todos los puntos encodeados


sns.set_theme()
sns.set()
sns.set(font_scale = 1.5)

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


plt.plot(xvals,yinterp,linewidth=3,color='orange', linestyle='dashed')

plt.savefig('encode_full_states.svg',dpi=300)


# hace una secuencia de matricas siguiendo el manifold

xvals_sub = np.linspace(UWS_centroid[0], W_centroid[0], 10)
yinterp_sub = np.interp(xvals_sub, x_cent, y_cent)


# Not Normalized
for i in range(10):
    z_sample = np.array([[xvals_sub[i], yinterp_sub[i]]])
    x_decoded = decoder.predict(z_sample)
    a_new=x_decoded[0].reshape(90,90)
    np. fill_diagonal(a_new, 1)
    plt.figure(figsize=(5,5))
    plt.imshow(a_new,cmap='magma')
    plt.xticks([0,10,20,30,40,50,60,70,80,90])
    plt.clim(-0.1, 1)
    plt.colorbar()
    plt.grid(None)
    plt.savefig('fcs_manifold'+str(i)+'.svg',dpi=300)



# Normalized
for i in range(10):
    z_sample = np.array([[xvals_sub[i], yinterp_sub[i]]])
    x_decoded = decoder.predict(z_sample)
    a=x_decoded[0].reshape(90,90)
    #np. fill_diagonal(a, 0)
    np. fill_diagonal(a, 1)
    a_new = (a-np.min(a))/(np.max(a)-np.min(a))  # between 0 and 1
    a_new = abs(a)/sum(sum(abs(a)))*10000  # equally sum    
    print(sum(sum(a_new)))
    plt.figure(figsize=(5,5))
    plt.imshow(a_new,cmap='magma')
    plt.xticks([0,10,20,30,40,50,60,70,80,90])
    plt.clim(0, 2)
    plt.grid(None)
    plt.savefig('fcs_manifold_norm'+str(i)+'.svg',dpi=300)















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
for kk in range(45):
    kick = np.transpose(kick_N1[:,:,kk])
    kick_data = (kick,label_kick)
    zz1 = plot_pert_in_latent(models,
             kick_data,'y','^',
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_N1 = np.asarray(zz)
pert_N1_centroid = np.mean(pert_N1,0)

kick_MCS = np.squeeze(kick_full_MCS.reshape(1,8100,20,45))
zz = []
for kk in range(45):
    kick = np.transpose(kick_MCS[:,:,kk])
    kick_data = (kick,label_kick)
    zz1=plot_pert_in_latent(models,
            kick_data,'m','x',
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
             kick_data,'r','x',
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
             kick_data,'r','*',
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
             kick_data,'y','^',
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
             kick_data,'r','^',
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
             kick_data,'g','x',
             batch_size=batch_size,
             model_name="vae_mlp")
    enco_pert =zz1[19,:]
    zz.append(enco_pert)
pert_sed= np.asarray(zz)
pert_sed_centroid = np.mean(pert_sed,0)