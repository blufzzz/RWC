import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import to_numpy

def one_epoch_pass(network, 
                   optimizer,
                   dataloader, 
                   metrics, 
                   loss_function, 
                   metric_function, 
                   train=True,
                   update_stats=True, 
                   imshow=False):
    
    CONTEXT = torch.enable_grad if train else torch.no_grad
    
    loss_history = []
    metric_history = []
    device = network.device
    
    for batch, target_batch in dataloader:

        inpt = batch.to(device)
        target_batch = target_batch.to(device)
        
        with CONTEXT():

            _, output = network(inpt, update_statistics=update_stats) # [[X, Y, Y_f],...,[X, Y, Y_f]], [d,batch_size]
            loss = loss_function(output.T, target_batch)
            
        if train:
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        target_pred = output.argmax(0)
        metric = metric_function(target_batch, target_pred)

        loss_history.append(loss.item())
        metric_history.append(metric)
    
    if imshow:
        plt.figure()
        plt.imshow(to_numpy(batch[-1].reshape(28,28)))
        plt.show()
        
    return loss_history, metric_history

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
