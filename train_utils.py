import numpy as np
import torch
import torch.nn as nn

def ls_loss(X,y,W,bias=False, onehot=True):
    y_oh = nn.functional.one_hot(y, num_classes=10).type(X.dtype)
    if not onehot:
        y_oh[y_oh == 0] = -1
    if bias:
        X_ = torch.cat([X, torch.ones_like(X[:,:1], 
                                           device=X.device, 
                                           dtype=X.dtype)], dim=-1)
    else:
        X_ = X.clone()
        
    loss = torch.norm(X_@W - y_oh, dim=-1)
    loss = torch.pow(loss, 2)
        
    return loss.mean()

def make_ls_solution(X,y,bias=False, onehot=True):
    '''
    X - tensor [N,d]
    y - tensor 
    '''
    y_oh = nn.functional.one_hot(y, num_classes=10).type(X.dtype)
    if not onehot:
        y_oh[y_oh == 0] = -1
    if bias:
        X_ = torch.cat([X, torch.ones_like(X[:,:1], 
                                           device=X.device, 
                                           dtype=X.dtype)], dim=-1)
    else:
        X_ = X.clone()
        
    S = X_.T@X_
    S += torch.randn_like(S)*1e-6 # for stability
    W_ls = torch.inverse(S)@X_.T@y_oh
    
    y = X_ @ W_ls
    
    return W_ls, y 



def get_grad_params(params):
    return list(filter(lambda x: x.requires_grad, params))
