import torch
from torch import nn
import numpy as np
from utils import to_numpy
from collections import defaultdict

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
        if self.add_bn:
            self.BN_s = []
            
        # for activation
        self.y_mean = torch.zeros(self.layers_number, self.hidden_dim, device=self.device)
        # for homeostasis
        self.y_H = torch.rand(self.layers_number, self.hidden_dim, device=self.device)*self.y_H_max
        
        # average activation history
        self.y_mean_s = defaultdict(list)
        self.delta_mean_s = defaultdict(list)
        
        
        # fill layers
        for layer in range(self.layers_number):
            
            out_dim = self.hidden_dim 
            in_dim = self.input_dim if layer==0 else self.hidden_dim
                
            # adding feed-forward connections
            self.W_s.append(nn.Parameter(torch.zeros(out_dim, in_dim), 
                                         requires_grad=self.W_requires_grad))
            
            self.b_s.append(nn.Parameter(torch.zeros(out_dim, 1), 
                                         requires_grad=self.b_requires_grad))
            
            if self.add_bn:
                self.BN_s.append(nn.BatchNorm1d(out_dim, affine=False, track_running_stats=False))
            
            # add activation function
            self.f_s.append(self.activation)
            
        # add readout for classes
        out_dim = out_dim + 1
        self.W_out = nn.Parameter(torch.zeros(self.n_classes, out_dim), 
                                  requires_grad=self.W_requires_grad)
        
        # create parameter lists
        self.W_s = nn.ParameterList(self.W_s)
        self.b_s = nn.ParameterList(self.b_s)
        self.f_s = nn.ModuleList(self.f_s)
            
        if self.add_bn:
            self.BN_s = nn.ModuleList(self.BN_s)
        
        self.init_weights()
        
    def init_weights(self, gaussian=True, mu=0, sigma=1):
        for pname,p in self.named_parameters():
            if pname.split('.')[0] == 'b_s' and not self.b_requires_grad:
                continue
            else:
                nn.init.xavier_normal_(p)

                
    def average_past_activity(self, T):
        
        self.y_mean_s[layer_number]
                

    def single_layer_forward(self, X, layer_number):
        
        '''
        Outputs results of consecutive operations in the layer
        X - input
        Y - output after synapses W layer
        Y_f - after nonlinearity f
        '''
        
        batch_size = X.shape[1]
        
        # normalize batch of [d,T] if needed
        if self.add_bn and batch_size > 1:
            # transpose to [T,d] for BS layer and back to [d,T]
            X = self.BN_s[layer_number](X.T).T 
        
        Y = self.W_s[layer_number]@X + self.b_s[layer_number] # [d,T]
        
        # homeostatis perturbation
        delta_y = self.delta_mult*(self.y_mean[layer_number] - self.y_H[layer_number]) # [d,1]
        
        if self.corruption:
            Y_f = self.f_s[layer_number](Y - delta_y.unsqueeze(1))
        else:
            Y_f = self.f_s[layer_number](Y)
            
        # update current average activity
        y_mean = Y_f.mean(1).detach()
        self.y_mean[layer_number] = (1-self.alpha)*y_mean + self.alpha*self.y_mean[layer_number] # 
        self.y_mean_s[layer_number].append(to_numpy(self.y_mean[layer_number]))
        self.delta_mean_s[layer_number].append(to_numpy(delta_y))
        
        return [X, Y, Y_f]
        
    def forward(self,X):
        
        '''
        X input batch - [T,d]
        '''
        
        batch_size, dim = X.shape
        X = X.T # to make it [d,T]
        layer_outputs = []
        
        for layer_number in range(self.layers_number):
            
            layer_output = self.single_layer_forward(X, layer_number)
            # activation to pass to the next layer
            # layer_output: [X, Y, Y_f]
            X = layer_output[-1]

            layer_outputs.append(layer_output)
            
        # single ouput readout
        X = torch.cat([X, torch.ones_like(X[:1,:], 
                                   device=X.device, 
                                   dtype=X.dtype)], dim=0)
            
        X = self.W_out@X
        

        return [layer_outputs, X]