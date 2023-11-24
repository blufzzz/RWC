import torch
from torch import nn
import numpy as np
from utils import to_numpy
from collections import defaultdict
from IPython.core.debugger import set_trace

class MLP(nn.Module):
    
    def __init__(self,**kwargs):
        
        super().__init__()
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        # initialize
#         if self.set_seed:
#             torch.manual_seed(self.seed)
            
        # for parameters
        self.W_s = []
        # for biases
        self.b_s = []
        # for activations
        self.f_s = []
        # for batch-normalization
        self.bn_s = []
            
        # for activation
        self.y_mean = torch.zeros(self.n_layers, self.hidden_dim, device=self.device)
        # for homeostasis
        self.y_H = torch.rand(self.n_layers, self.hidden_dim, device=self.device)*self.y_H_max
        
        self.y_mean = self.y_H.clone()
        
        # average activation history
        self.y_mean_s = defaultdict(list)
        self.delta_mean_s = defaultdict(list)
        
        
        # fill layers
        for layer in range(self.n_layers):
            
            out_dim = self.hidden_dim 
            in_dim = self.input_dim if layer==0 else self.hidden_dim
                
            # adding feed-forward connections
            self.W_s.append(nn.Parameter(torch.zeros(out_dim, in_dim), 
                                         requires_grad=self.W_requires_grad))
            
            self.b_s.append(nn.Parameter(torch.zeros(out_dim, 1), 
                                         requires_grad=self.b_requires_grad))
            
            self.bn_s.append(nn.BatchNorm1d(in_dim, affine=self.bn_affine, track_running_stats=False))
            
            # add activation function
            self.f_s.append(self.activation)

        # before readout
        self.bn_s.append(nn.BatchNorm1d(out_dim, affine=self.bn_affine, track_running_stats=False))
            
        # add readout for classes
        out_dim = out_dim + 1
        self.W_out = nn.Parameter(torch.zeros(self.n_classes, out_dim), 
                                  requires_grad=self.W_requires_grad)
        
        # create parameter lists
        self.W_s = nn.ParameterList(self.W_s)
        self.b_s = nn.ParameterList(self.b_s)
        self.f_s = nn.ModuleList(self.f_s)
        self.bn_s = nn.ModuleList(self.bn_s)
        
        self.init_weights()
        
        
    def batch_normalization(self, X, layer_number):
        
        '''
        X - [bs,d]
        '''
        
        batch_size = X.shape[1]
        
        # normalize batch of [d,T] if needed
        if self.use_bn and batch_size > 1:
            # transpose to [T,d] for BS layer and back to [d,T]
            X = self.bn_s[layer_number](X.T).T 
            
        return X
        
        
    def clear_statistics(self, update_history=False):
        
        
        if update_history:
            if hasattr(self, 'y_mean_s_history'):
                for layer_number in range(self.n_layers):
                    self.y_mean_s_history[layer_number] = self.y_mean_s_history[layer_number] +\
                                                            self.y_mean_s[layer_number]
            else:
                self.y_mean_s_history = self.y_mean_s


            if hasattr(self, 'delta_mean_s_history'):
                for layer_number in range(self.n_layers):
                    self.delta_mean_s_history[layer_number] = self.delta_mean_s_history[layer_number] +\
                                                                self.delta_mean_s[layer_number]
            else:
                self.delta_mean_s_history = self.delta_mean_s
            
        # new default parameters
        self.y_mean = self.y_H.clone() # cloning is essential!
        self.y_mean_s = defaultdict(list)
        self.delta_mean_s = defaultdict(list)
        
    def restore_statistics(self):
        
        self.y_mean = self.y_mean_history
        self.y_mean_s = self.y_mean_s_history
        self.delta_mean_s = self.delta_mean_s_history
        
    def init_weights(self, gaussian=True, mu=0, sigma=1):
        for pname,p in self.named_parameters():
            if pname.split('.')[0] == 'bn_s':
                continue
            if pname.split('.')[0] == 'b_s' and not self.b_requires_grad:
                continue
            else:
                nn.init.xavier_normal_(p)


    def single_layer_forward(self, X, layer_number, update_statistics=True):
        
        '''
        Outputs results of consecutive operations in the layer
        X - input
        Y - output after synapses W layer
        Y_f - after nonlinearity f
        '''
        
        
        X = self.batch_normalization(X, layer_number)
        X = self.W_s[layer_number]@X + self.b_s[layer_number] # [d,T]
        
        # homeostatis perturbation - batch-independent!
        delta_y = self.delta_mult*(self.y_mean[layer_number] - self.y_H[layer_number]) # [d,1]
        
        if self.corruption:
            X = self.activation(X - delta_y.unsqueeze(1)) # 
        else:
            X = self.activation(X)
            
        # update current average activity
        if update_statistics:
#             print(layer_number, self.y_mean[layer_number].mean(), self.y_H[layer_number].mean())
            y_mean = X.mean(1).detach()
            self.y_mean[layer_number] = (1-self.alpha)*y_mean + self.alpha*self.y_mean[layer_number] 
            self.y_mean_s[layer_number].append(to_numpy(self.y_mean[layer_number]))
            self.delta_mean_s[layer_number].append(to_numpy(delta_y))
        
        return X
        
    def forward(self,X, update_statistics=True):
        
        '''
        X input batch - [T,d]
        '''
        
        batch_size, dim = X.shape
        X = X.T # to make it [d,T]
        layer_outputs = []
        
        for layer_number in range(self.n_layers):
            
            # activation to pass to the next layer
            # layer_output: [X, Y, Y_f]
            X = self.single_layer_forward(X, layer_number, update_statistics)
            layer_outputs.append(X.detach())
        
        # `before-readout` normalization
        X = self.batch_normalization(X, -1)
        
        # single ouput readout
        Y = torch.cat([X, torch.ones_like(X[:1,:], 
                                   device=X.device, 
                                   dtype=X.dtype)], dim=0)
        
        Y = self.W_out@Y

        return [layer_outputs, Y]