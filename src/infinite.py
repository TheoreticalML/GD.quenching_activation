from copy import deepcopy
import math
import numpy as np 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
cpu = torch.device("cpu")


def get_target(d, target, m_target=10):
    if target == 'one-neuron':
        a_st = torch.tensor([1])
        B_st = torch.zeros(1,d)
        B_st[0,0] = 1
    elif target == 'linear':
        a_st = torch.tensor([1,-1])
        B_st = torch.zeros(2,d)
        B_st[0,0] = 1
        B_st[1,0] = -1
    elif target == 'circle':
        M = 100
        a_st = torch.ones(M)/M
        B_st = torch.zeros(M,d)
        for i in range(M):
            B_st[i,0] = math.cos(2*math.pi*i/M)
            B_st[i,1] = math.sin(2*math.pi*i/M)
    elif target == 'multi-neuron':
        M = m_target
        a_st = torch.ones(M)/M
        B_st = torch.randn(M,d)
        k = int(d)
        B_st[:,k:]=0
        B_st /= B_st.norm(dim=1, keepdim=True)
    else:
        ValueError('Target: {} is not supported'.format(target))
    return a_st, B_st


def get_kernel_martix(B1, B2, EPS=1e-12):
    n1 = B1.norm(dim=1, keepdim=True)
    n2 = B2.norm(dim=1, keepdim=True)
    v1 = B1/n1
    v2 = B2/n2

    n1_o_n2 = n1.matmul(n2.t())
    B1_o_B2 = B1.matmul(B2.t())
    v1_o_v2 = torch.clamp(v1.matmul(v2.t()), -1+EPS, 1-EPS)

    K = torch.sqrt(n1_o_n2**2-B1_o_B2**2+EPS) +  B1_o_B2*(math.pi-torch.acos(v1_o_v2))
    return K



class MultiNeuron(nn.Module):
    def __init__(self, m, d, EPS=1e-12, rf=False, mf=False):
        super(MultiNeuron, self).__init__()

        a = torch.zeros(m)
        B = torch.randn(m, d)
        B = B/B.norm(dim=1, keepdim=True)

        self.a = nn.Parameter(a.double().to(device))
        self.B = nn.Parameter(B.double().to(device))

        self.d = d
        self.m = m
        self.EPS = EPS
        self.rf = rf
        self.mf = mf
        
        if rf:
            self.B.requires_grad=False


    def set_target(self, a_st, B_st):
        self.B_st = B_st.double().to(device)
        self.a_st = a_st.double().to(device)
        self.K3 = get_kernel_martix(self.B_st, self.B_st, self.EPS)
        self.y3 = torch.matmul(self.a_st, self.K3.matmul(self.a_st))

    def solve_rand_feature(self):
        EPS = self.EPS
        K1 = get_kernel_martix(self.B, self.B, EPS).data
        K2 = get_kernel_martix(self.B, self.B_st, EPS).data

        y = K2.matmul(self.a_st).view(-1,1).data
        a, _ = torch.solve(y, K1)
        self.a.data.copy_(a.squeeze())

        return a.squeeze()


    def forward(self):
        K1 = get_kernel_martix(self.B, self.B, self.EPS)
        K2 = get_kernel_martix(self.B, self.B_st, self.EPS)
        K3 = self.K3

        y1 = torch.matmul(self.a, K1.matmul(self.a))
        y2 = torch.matmul(self.a, K2.matmul(self.a_st))
        y3 = self.y3

        if self.mf:
            y1 /= self.m*self.m
            y2 /= self.m
        y = y1 - 2*y2 + y3

        return y


class Train:
    def __init__(self, m, d, target='one-neuron', rf=False, mf=False, m_target=10):
        self.nepoch_r = []
        self.loss_r = []
        self.a_r = []
        self.B_r = []
 
        self.m = m
        self.d = d
 
        self.net = MultiNeuron(m, d, rf=rf, mf=mf)
        a_st, B_st = get_target(d, target, m_target)
        self.net.set_target(a_st, B_st)


    def run(self, nepochs=50000, learning_rate=5e-3, TOL=1e-5, 
            plot_epoch=1000, check_epoch=-1, solver='gd'):
        if solver == 'gd':
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)
        elif solver == 'momentum':
            optimizer = optim.SGD(self.net.parameters(),lr=learning_rate, momentum=0.9)
        elif solver == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            raise ValueError('The optimizer {} is not supported'.format(solver))

        for epoch in range(nepochs):
            optimizer.zero_grad()
            loss = self.net()
            loss.backward()
            optimizer.step()


            if epoch%plot_epoch == 0:
                print('[{:}/{:}], {:.1e}'.format(epoch+1, nepochs, loss.item()))
            if loss.item() < TOL:
                break

            if check_epoch>0 and epoch%check_epoch == 0:
                a = self.net.a.data.to(cpu).clone().tolist()
                B = self.net.B.data.to(cpu).clone().tolist()

                self.a_r.append(a)
                self.B_r.append(B)
                self.nepoch_r.append(epoch+1)
                self.loss_r.append(loss.item())