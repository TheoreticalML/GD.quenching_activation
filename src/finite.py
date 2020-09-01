import math
import numpy as np 
from scipy.linalg import svd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
cpu = torch.device("cpu")


def minimum_norm_solver(A, b):
    """
    input: A 
            minimize ||x||_2 
                    s.t. Ax=b
    """
    U, s, Vh = svd(A, full_matrices=False, lapack_driver='gesvd')
    b_proj = np.matmul(U.transpose(), b)
    z = b_proj / s 
    x = np.matmul(Vh.transpose(), z)
    return x

def get_data(n_tr, n_te, d, target='one-neuron', m_target=10):
    x_tr = torch.randn(n_tr, d).to(device)
    x_te = torch.randn(n_te, d).to(device)

    if target == 'one-neuron':
        y_tr = (x_tr[:,0]>0).float() * x_tr[:,0]
        y_te = (x_te[:,0]>0).float() * x_te[:,0]

    elif target =='linear':
        y_tr = x_tr[:,0]
        y_te = x_te[:,0]

    elif target == 'multi-neuron':
        m = m_target
        w = torch.randn(m, d)
        w[:,10:] = 0
        w /= w.norm(dim=1, keepdim=True)

        f = x_tr.matmul(w.t())
        y_tr = F.relu(f).mean(dim=1)

        f = x_te.matmul(w.t())
        y_te = F.relu(f).mean(dim=1)

    elif target == 'circle':
        m = 1000
        net = Network(d, m)
        net.a.data.fill_(1/m)
        net.B.data = torch.zeros(m,d).to(device)
        for i in range(m):
            net.B.data[i,0] = math.cos(2*math.pi*i/m)
            net.B.data[i,1] = math.sin(2*math.pi*i/m)

        y_tr = net(x_tr).data
        y_te = net(x_te).data

    return x_tr.to(device), y_tr.to(device), x_te.to(device), y_te.to(device)



class Network(nn.Module):
    def __init__(self, d, m, rf=False, mf=False):
        super(Network, self).__init__()

        a = torch.zeros(m)
        B = torch.randn(m,d)
        B /= B.norm(dim=1, keepdim=True)

        self.a = nn.Parameter(a.to(device))
        self.B = nn.Parameter(B.to(device))

        self.m = m
        self.rf = rf
        self.mf = mf
        if rf:
            self.B.requires_grad = False


    def forward(self, x):
        o = x.matmul(self.B.t())
        o = F.relu(o)
        o = o.matmul(self.a)

        if self.mf:
            o /= self.m

        return o.squeeze()

    def feature(self, x):
        Bx = x.matmul(self.B.t())
        return F.relu(Bx)

    def path_norm(self):
        pn =  self.a * self.B.norm(dim=1)
        pn = pn.data.abs().sum().item()
        return pn 



class Train:
    def __init__(self, m, n, d, n_te=10000,
            target='one-neuron', m_target=10, rf=False, mf=False, save_aB=False):
        self.m = m
        self.n = n
        self.d = d
        self.net = Network(d, m, rf=rf, mf=mf)
        self.x_tr, self.y_tr, self.x_te, self.y_te = get_data(n, n_te, d, target, m_target)

        self.loss_tr_r = []
        self.loss_te_r = []
        self.nepoch_r = []
        self.pnorm_r = []

        self.save_aB = save_aB
        if save_aB:
            self.a_r = []
            self.B_r = []

    def run(self, nepochs=1000, learning_rate=5e-3, TOL=1e-5,
            plot_epoch=500, check_epoch=-1, solver='gd', batch_size=-1):
        """
        if batch_size != -1, mini-batch are used
        """
        n_tr = len(self.y_tr)
        if batch_size == -1:
            batch_size = n_tr
        if solver == 'gd':
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)
        elif solver == 'momentum':
            optimizer = optim.SGD(self.net.parameters(),lr=learning_rate, momentum=0.9)
        elif solver == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        else:
            raise ValueError('the optimizer {} is not supported'.format(solver))


        for epoch in range(nepochs):
            loss_tot, nbatch = 0, 0
            for i in range(0, n_tr, batch_size):
                j = min(i+batch_size, n_tr)
                x_b, y_b = self.x_tr[i:j, :], self.y_tr[i:j]
                
                optimizer.zero_grad()
                y_p = self.net(x_b)
                loss = (y_p - y_b).pow(2).mean()
                loss.backward()
                optimizer.step()
                nbatch += 1
                loss_tot += loss.item()

            if epoch%plot_epoch == 0:
                print('{:}-th iter, loss: {:.1e}'.format(epoch, loss_tot/nbatch))

            if check_epoch>=1 and epoch%check_epoch==0:
                self.loss_tr_r.append(loss_tot/nbatch)
                self.loss_te_r.append(self.validate())
                self.pnorm_r.append(self.net.path_norm())
                self.nepoch_r.append(epoch+1)

                if self.save_aB:
                    a = self.net.a.data.to(cpu).clone().tolist()
                    B = self.net.B.data.to(cpu).clone().tolist()

                    self.a_r.append(a)
                    self.B_r.append(B)

            if loss_tot/nbatch < TOL:
                break

    def validate(self, batch_size=100):
        n_te = self.y_te.shape[0]
        e_tot = 0
        nbatch = 0
        for i in range(0, n_te, batch_size):
            j = min(i+batch_size, n_te)
            bx, by = self.x_te[i:j], self.y_te[i:j]

            y_pre = self.net(bx).squeeze().data
            E = (y_pre - by).pow(2).mean()
            e_tot += E.item()
            nbatch += 1
        return e_tot/nbatch

    def solve_rand_feature(self):
        F_np = self.net.feature(self.x_tr).data.numpy()
        y_np = self.y_tr.numpy()

        a_np = minimum_norm_solver(F_np, y_np)
        a = torch.from_numpy(a_np)

        self.net.a.data.copy_(a)
            
        return a


if __name__ == '__main__':
    m = 10
    n = 400
    d = 20

    nn = Train(m, n, d, target='one-neuron')
    nn.run(nepochs=10000, learning_rate=5e-3, plot_epoch=100)
    print(nn.net.a.data)
    

    
