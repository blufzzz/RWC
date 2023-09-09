import numpy as np
import tqdm
import torch
from torch import nn
from collections import defaultdict
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib as mpl
from IPython.core.debugger import set_trace



def r2_score_batch(label_all, Y_all):
    y_true = to_numpy(label_all)
    y_pred = to_numpy(Y_all)
    if y_true.ndim == 2:
        assert y_pred.ndim == 2
        # batch-mode
        return np.mean([r2_score(y_true_i, y_pred_i) for y_true_i, y_pred_i in zip(y_true, y_pred)])
    else:
        return r2_score(y_true, y_pred)

def MSE_smoothed(Y_all, label_all, kernel_size=10):
    
    if label_all.ndim == 1:
        label_all = label_all.unsqueeze(0)
       
    if Y_all.ndim == 1:
        Y_all = Y_all.unsqueeze(0)
    
    # do the smoothing
    if kernel_size is not None:
        kernel_size = kernel_size+1 if kernel_size%2 == 0 else kernel_size
        smooth_kernel = torch.tensor(windows.blackman(kernel_size), 
                                     dtype=label_all.dtype).unsqueeze(0).unsqueeze(1).to(device)
        
        label_all_conv = torch.conv1d(label_all.unsqueeze(1), # add fake channel dimension 
                                      smooth_kernel, padding=kernel_size//2)[0,0]
    else:
        label_all_conv = label_all
    
    return torch.norm(Y_all - label_all_conv, dim=-1).mean(0)


def get_cmap_color(i, N, cmap='viridis', N_CMAP=256):
    c_i = int(N_CMAP*(i/N))
    return mpl.colormaps[cmap].colors[c_i]

def settings(ax=None):
    global DPI
    DPI=500
    plt.rcParams['font.size'] = 40
    plt.rcParams['axes.linewidth'] = 4
    
    if ax is not None:
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)


def normalize_np(data):
    return (data - data.mean(0,keepdims=True))/(data.std(0,keepdims=True)+1e-7)

def minmax(data):
    return (data - data.min()) / (data.max() - data.min())


def get_capacity(model):
    s_total = 0
    for param in model.parameters():
        s_total+=param.numel()
    return round(s_total,2)


def calc_gradient_norm(parameters):
    total_norm = 0.0
    for i,p in enumerate(parameters):
        try:
            param_norm = p.grad.data.norm(2)
        except:
            set_trace()
        total_norm += param_norm.item()
    total_norm = total_norm 
    return np.mean(total_norm)


def create_xavier_W_param(d1,d2,requires_grad=False):
    W = torch.randn(d1,d2)
    W = nn.Parameter(W, requires_grad=requires_grad)
    nn.init.xavier_normal_(W)
    return W

def clone_parameter(W):
    W_new = W.detach().clone()
    W_new.requires_grad_(True)
    return W_new

def bn(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)

def to_numpy(x):
    return x.detach().cpu().numpy()



def gs_orthogonalization(*args):
    
    v = args
    d = len(v)
    
    v_0 = v[0]
    u_0 = v_0 / torch.norm(v_0)
    u_orth = [u_0]
    
    for i in range(1,d):
        u_i = v[i]
        for u in u_orth:
            u_i -= (v[i].T@u)*u
        u_i = u_i / torch.norm(u_i)
        u_orth.append(u_i)
        
    return u_orth

def accuracy(y, y_pred):
    return accuracy_score(to_numpy(y), to_numpy(y_pred))

def cosine_similarities(X_0, X_s):
    sims = []
    for X in X_s:
        sim = (X.flatten()@X_0.flatten()).item()
        sims.append(sim)
    return np.array(sims)