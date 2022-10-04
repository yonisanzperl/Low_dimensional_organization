# -*- coding: utf-8 -*-
"""
Created on Fri May  6 05:41:57 2022

@author: yonis
"""

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



import scipy.io

mat = scipy.io.loadmat('stat_brainstates_to_plot.mat')
means_s=mat['means']
QS_s=mat['Qs']
CorrSC_s = mat['corrstruct']

mat = scipy.io.loadmat('stat_multiverse_to_plot.mat')
means_mv=mat['means']
QS_mv=mat['Qs']
CorrSC_mv = mat['corrstruct']

states = [0, 1 ,2 ,3 ,4 ,5 ,6 ,7]
# vs brainstates
fig, axes = plt.subplots(ncols=3, nrows=1,figsize=(15,5))
plt.subplot(131)
plt.plot(states,np.mean(means_s,1), 'yellowgreen', label='mean FC')
plt.fill_between(states, np.mean(means_s,1) - np.std(means_s,1),np.mean(means_s,1) + np.std(means_s,1), color='yellowgreen', alpha=0.2)
plt.xlabel('State',fontsize=20)
plt.ylabel('mean FC',fontsize=20,labelpad = 15)
#plt.legend(fontsize=10)

plt.subplot(132)
plt.plot(states,np.mean(QS_s,1), 'orange', label='mean FC')
plt.fill_between(states, np.mean(QS_s,1) - np.std(QS_s,1),np.mean(QS_s,1) + np.std(QS_s,1), color='orange', alpha=0.2)
plt.xlabel('State',fontsize=20)
plt.ylabel('Qs FC',fontsize=20,labelpad = 15)
#plt.legend(fontsize=10)

plt.subplot(133)
plt.plot(states,np.mean(CorrSC_s,1), 'purple', label='mean FC')
plt.fill_between(states, np.mean(CorrSC_s,1) - np.std(CorrSC_s,1),np.mean(CorrSC_s,1) + np.std(CorrSC_s,1), color='purple', alpha=0.2)
plt.xlabel('State',fontsize=20)
plt.ylabel('Corr SC',fontsize=20,labelpad = 15)
#plt.legend(fontsize=10)

plt.savefig('stat_brainstates.svg', format='svg', dpi=300)
# vs multiverse

cmap2 = 'Greens'
fig, axes = plt.subplots(ncols=3, nrows=1,figsize=(15,5))
plt.subplot(131)
ax = sns.heatmap(means_mv.reshape(20,20),xticklabels= False,yticklabels=False,cmap= cmap2,cbar = True, vmin= 0.0, vmax = 0.7)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

cmap2 = 'Oranges'
plt.subplot(132)
ax = sns.heatmap(QS_mv.reshape(20,20),xticklabels= False,yticklabels=False,cmap= cmap2,cbar = True, vmin= 0.0, vmax = 0.4)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

cmap2 = 'BuPu'
#cmap ='Purples'
plt.subplot(133)
ax = sns.heatmap(CorrSC_mv.reshape(20,20),xticklabels= False,yticklabels=False,cmap= cmap2,cbar = True, vmin= 0.0, vmax = 0.7)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#plt.savefig('stat_multi.svg', format='svg', dpi=300)



mat = scipy.io.loadmat('stats_perturbaciones.mat')
means_N1=mat['means_N1']
means_N2=mat['means_N2']
means_N3=mat['means_N3']
means_sed=mat['means_sed']
means_Inc=mat['means_Inc']
means_MCS=mat['means_MCS']
means_UWS=mat['means_UWS']
Qs_N1=mat['Qs_N1']
Qs_N2=mat['Qs_N2']
Qs_N3=mat['Qs_N3']
Qs_sed=mat['Qs_sed']
Qs_Inc=mat['Qs_Inc']
Qs_MCS=mat['Qs_MCS']
Qs_UWS=mat['Qs_UWS']

CorrSC_N1 = mat['corrstruct_N1']
CorrSC_N2 = mat['corrstruct_N2']
CorrSC_N3 = mat['corrstruct_N3']
CorrSC_sed = mat['corrstruct_sed']
CorrSC_Inc = mat['corrstruct_Inc']
CorrSC_MCS = mat['corrstruct_MCS']
CorrSC_UWS = mat['corrstruct_UWS']

# vs pertu stats
fig, axes = plt.subplots(ncols=3, nrows=1,figsize=(15,5))
plt.subplot(131)
plt.plot(np.mean(means_N1,1),color='purple',label='N1')
plt.plot(np.mean(means_N2,1),color='yellow',label='N2')
plt.plot(np.mean(means_N3,1), color='orange',label='N3')
plt.plot(np.mean(means_sed,1),color='g', label='Sed')
plt.plot(np.mean(means_Inc,1),color='r', label='Inc')
plt.plot(np.mean(means_MCS,1),color='tan',label='MCS')
plt.plot(np.mean(means_UWS,1),color='c',label='UWS')
plt.legend(fontsize=10)

plt.subplot(132)
plt.plot(np.mean(Qs_N1,1),color='purple')
plt.plot(np.mean(Qs_N2,1),color='yellow')
plt.plot(np.mean(Qs_N3,1),color='orange')
plt.plot(np.mean(Qs_sed,1),color='g')
plt.plot(np.mean(Qs_Inc,1),color='r')
plt.plot(np.mean(Qs_MCS,1),color='tan')
plt.plot(np.mean(Qs_UWS,1),color='c')

plt.subplot(133)
plt.plot(np.mean(CorrSC_N1,1),color='purple')
plt.plot(np.mean(CorrSC_N2,1),color='yellow')
plt.plot(np.mean(CorrSC_N3,1),color='orange')
plt.plot(np.mean(CorrSC_sed,1),color='g')
plt.plot(np.mean(CorrSC_Inc,1),color='r')
plt.plot(np.mean(CorrSC_MCS,1),color='tan')
plt.plot(np.mean(CorrSC_UWS,1),color='c')

plt.savefig('stat_pertu.svg', format='svg', dpi=300)

# vs distacne stats (se calculan en el Figure3.py al final)
fig, axes = plt.subplots(ncols=3, nrows=1,figsize=(15,5))
plt.subplot(131)
plt.plot(N1_dist_mean,color='purple',label='N1')
plt.plot(N2_dist_mean,color='yellow',label='N2')
plt.plot(N3_dist_mean, color='orange',label='N3')
plt.plot(sed_dist_mean,color='g', label='Sed')
plt.plot(Inc_dist_mean,color='r', label='LoC')
plt.plot(MCS_dist_mean,color='tan',label='MCS')
plt.plot(UWS_dist_mean,color='c',label='UWS')
plt.legend(fontsize=10)

plt.subplot(132)
plt.plot(N1_dist_to_W_mean,color='purple')
plt.plot(N2_dist_to_W_mean,color='yellow')
plt.plot(N3_dist_to_W_mean,color='orange')
plt.plot(sed_dist_to_W_mean,color='g')
plt.plot(Inc_dist_to_W_mean,color='r')
plt.plot(MCS_dist_to_W_mean,color='tan')
plt.plot(UWS_dist_to_W_mean,color='c')

plt.savefig('stat_pertu_distances.svg', format='svg', dpi=300)