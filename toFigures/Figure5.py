from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from numpy import genfromtxt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


from skimage.metrics import structural_similarity as ssim
import seaborn as sns
import pandas as pd
import os


def lower_tri_masking_mat(A):
    mat_mit = np.zeros((35,35))
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] > r
    mat_mit[mask] = A[mask]
    return mat_mit

matrix = genfromtxt('all_info.csv',delimiter=',')
ss = genfromtxt('significant.csv',delimiter=',')
x_axis_labels = ['Dist. to W','Dist. to Ori','Mean FC','Net. Mod.','FC-SC coupl.','Dist. to W','Dist. to Ori','Mean FC','Net. Mod.','FC-SC coupl.','Dist. to W','Dist. to Ori','Mean FC','Net. Mod.','FC-SC coupl.','Dist. to W','Dist. to Ori','Mean FC','Net. Mod.','FC-SC coupl.','Dist. to W','Dist. to Ori','Mean FC','Net. Mod.','FC-SC coupl.','Dist. to W','Dist. to Ori','Mean FC','Net. Mod.','FC-SC coupl.','Dist. to W','Dist. to Ori','Mean FC','Net. Mod.','FC-SC coupl.'] # labels for x-axis


corres = np.corrcoef(matrix.T)
#low_matrix=lower_tri_masking_mat(corres)


plt.figure(figsize=(12,10))
a=sns.heatmap(corres,yticklabels=x_axis_labels,cmap='bwr')
a.set(xticklabels=[])
a = plt.scatter(ss[:,0]-0.5,ss[:,1]-0.5,s=10, marker='*')




plt.savefig('CorrMatrix2_sig.svg', format='svg', dpi=200)