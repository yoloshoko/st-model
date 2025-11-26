'''
STGCN
'''
import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from torchsummary import summary

'''
align主要是对数据格式进行一个处理,类似于reshape
'''

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:  # b,c,t,n -> b,c_out,t,n  对c维度进行右零填充
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x
'''
门控卷积单元的定义,目的是用来提取时空特征
GLU和sigmoid
'''
class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):  # 3,2,16,"GLU"
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):   # b,c,t,n
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # b,c_out,t,n -> b,c_out,[t-kt+1],n
        if self.act == "GLU":
            x_conv = self.conv(x)  # b,c,t,n -> b,2*c_out,[t-kt+1],n
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])  # b,c_out,[t-kt+1],n
        if self.act == "sigmoid":   # b,c,t,n -> b,c_out,[t-kt+1],n
            return torch.sigmoid(self.conv(x) + x_in)  # b,c_out,[t-kt+1],n 
        return torch.relu(self.conv(x) + x_in)
'''
空间卷积层
'''

class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):   # 3,16,(3,50,50)
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))  # h,h,s
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))    # 1,h,1,1
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # gp.quicksum( lk[k,n,m] * x[b,i,t,m] for m in range(M) )
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)     # b,h,k,n * s,n,n = b,h,k,s,n
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # h,h,s * b,h,k,s,n + 1,h,1,1 = b,h,k,n + 1,h,1,1 = b,h,k,n
        return torch.relu(x_gc + x)  # b,h,k,n

'''
这就是对应的一个ST_conv模块,会调用到前面定义好的tconv和sconv
'''

class st_conv_block(nn.Module):
    def __init__(self, ks, kt, n, c, p, Lk):  # 3,3,50,[2, 16, 64],0,(3,50,50)
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")  # 3,2,16,"GLU"
        self.sconv = spatio_conv_layer(ks, c[1], Lk)  # 3,16,(3,50,50)
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2]) # 3,16,64
        self.ln = nn.LayerNorm([n, c[2]])  # 50,64

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_t1 = self.tconv1(x)   # b,c_1,[t-kt+1],n
        x_s = self.sconv(x_t1)  # b,c_1,[t-kt+1],n
        x_t2 = self.tconv2(x_s) # b,c_2,[t-2*kt+2],n
        # print(x_t1.shape,x_s.shape,x_t2.shape)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # # b,[t-2*kt+2],n，c_2   
        return self.dropout(x_ln)

class output_layer(nn.Module):
    def __init__(self, c, T, n):  # c,k,n
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = nn.Conv2d(c, 1, 1)

    def forward(self, x):  # b,c,k,n
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        # print(x.shape,x_t1.shape,x_t2.shape)
        return self.fc(x_t2)

class STGCN(nn.Module):  
    def __init__(self, ks, kt, bs, T, n, Lk, p):   # 3, 3, [[2, 16, 64], [64, 16, 64]], 12, 50, (3,50,50) ,0
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)  # 3,3,50,[2, 16, 64],0,(3,50,50)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)  # 3,3,50,[64, 16, 64],0,(3,50,50)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1), n)  # c=64,k,n

    def forward(self, x):   
        x_st1 = self.st_conv1(x)      # b,c,k,n
        x_st2 = self.st_conv2(x_st1)  # b,c,k,n
        # print(x_st1.shape,x_st2.shape)
        return self.output(x_st2)

def weight_matrix(W, sigma2=0.1, epsilon=0.5):  # 邻接矩阵归一化
    '''
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    n = W.shape[0]
    W = W /10000
    W[W==0]=np.inf  # 关联为0的最后趋近与0
    W2 = W * W
    W_mask = (np.ones([n, n]) - np.identity(n))
    return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

def scaled_laplacian(A):  # 归一化拉普拉斯矩阵
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)   # 图结构表示

def cheb_poly(L, Ks):    # 切比雪夫多项式
    n = L.shape[0] 
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)

def main():
    CHANNEL= 2
    TIMESTEP_IN = 12
    N_NODE = 50
    ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
    A = np.random.randn(50,50)
    W = weight_matrix(A)
    L = scaled_laplacian(W)
    Lk = cheb_poly(L, ks)
    Lk = torch.Tensor(Lk.astype(np.float32))

    x = torch.randn(64,2,12,50)  # b,c,t,n
    model = STGCN(ks, kt, bs, T, n, Lk, p)
    y = model(x)
    print(W.shape,L.shape)  # n, n and n, n
    print(Lk.shape)         # ks,n,n
    print(x.shape,y.shape) 
if __name__ == '__main__':
    main()