import numpy as np
import torch
import matplotlib.pyplot as plt 

def plot_dynamics(nn, label='finite'):
    fig, axes = plt.subplots(1,3,figsize=(14,3))
    if label=='finite':
        axes[0].semilogy(nn.nepoch_r, nn.loss_tr_r,'-', label='train')
        axes[0].semilogy(nn.nepoch_r, nn.loss_te_r,'-', label='test')
    else:
        axes[0].semilogy(nn.nepoch_r, nn.loss_r, '-', label='loss')
    axes[0].legend()
    axes[0].set_title('loss')

    # loss 
    a_r = torch.tensor(nn.a_r)
    B_r = torch.tensor(nn.B_r)
    neuron_magnitude = a_r * B_r.norm(dim=2)
    for i in range(nn.m):
        axes[1].plot(nn.nepoch_r, neuron_magnitude[:,i])
    axes[1].set_title('dynamics of neuron magnitude')

    # 
    axes[2].plot(neuron_magnitude[-1,:],'o')
    axes[2].set_title('final solution')