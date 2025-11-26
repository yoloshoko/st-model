import torch
from torch import nn 
import numpy as np
from torch.nn import functional as F

class align(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__()
        self.c_in = c_in 
        self.c_out = c_out 
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in,c_out,1)
    def forward(self,x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x,[0,0,0,0,0,self.c_out-self.c_in,0,0])
        return x

class temporal_conv_layer(nn.Module):
    def __init__(self,kt=3,c_in=16,c_out=16,act="relu"):
        super().__init__()
        self.kt = kt 
        self.act = act
        self.c_out = c_out
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in,c_out*2,(kt,1),1)
        else:
            self.conv = nn.Conv2d(c_in,c_out,(kt,1),1)
        self.align = align(c_in,c_out)
    def forward(self,x):
        x_in = self.align(x)[:,:,self.kt-1:,:]
        if self.act == "GLU":
            x_conv = self.conv(x)
            y = x_conv[:,:self.c_out,:,:] + x_in
            weight = torch.sigmoid(x_conv[:,self.c_out:,:,:])
            return y * weight
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)
    
class spatio_conv_layer(nn.Module):
    def __init__(self,ks=3,c=16,lk=torch.randn(3,50,50)):
        super().__init__()
        self.lk = lk 
        self.theta = nn.Parameter(torch.FloatTensor(c,c,ks))
        self.b = nn.Parameter(torch.FloatTensor(1,c,1,1))
    def forward(self,x):
        x_c = torch.einsum("knm,bitm->bitkn",self.lk,x)
        x_gc = torch.einsum("iok,bitkn->botn",self.theta,x_c) +self.b
        return torch.relu(x_gc + x)

class st_conv_block(nn.Module):
    def __init__(self,ks=3,kt=3,n=50,c=[2,16,64],lk=torch.randn(3,50,50)):
        super().__init__()
        self.tconv1 = temporal_conv_layer(kt,c[0],c[1],"GLU")
        self.sconv = spatio_conv_layer(ks,c[1],lk)
        self.tconv2 = temporal_conv_layer(kt,c[1],c[2])
        self.ln = nn.LayerNorm([n,c[2]])
        self.dropout = nn.Dropout(p=0)
    def forward(self,x):
        x_t1 = self.tconv1(x)
        x_s = self.sconv(x_t1)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0,2,3,1)).permute(0,3,1,2)
        return self.dropout(x_ln)
    
class output_layer(nn.Module):
    def __init__(self,c=64,t=4,n=50):
        super().__init__()
        self.t_conv1 = temporal_conv_layer(t,c,c,"GLU")
        self.ln = nn.LayerNorm([n,c])
        self.t_conv2 = temporal_conv_layer(1,c,c,"sigmoid")
        self.fc = nn.Conv2d(c,1,1)
    def forward(self,x):
        x_t1 = self.t_conv1(x)
        x_ln = self.ln(x_t1.permute(0,2,3,1)).permute(0,3,1,2)
        x_t2 = self.t_conv2(x_ln)
        return self.fc(x_t2)

class stgcn(nn.Module):
    def __init__(self,ks,kt,bs,lk,t=12,pred_num=6):
        super().__init__()
        self.n = lk.shape[-1]
        self.st_conv1 = st_conv_block(ks,kt,self.n,bs[0],lk)
        self.st_conv2 = st_conv_block(ks,kt,self.n,bs[1],lk)
        self.output = output_layer(bs[1][2],t - 4*(kt - 1),self.n)
        self.fc = nn.Linear(1,pred_num)
    def forward(self,x):  
        x = x.permute(0,3,2,1)
        x_st1 = self.st_conv1(x)
        x_st2 = self.st_conv2(x_st1)
        out = self.output(x_st2)
        out = out.permute(0,3,1,2).squeeze(-1)
        out = self.fc(out)
        return out

def weight_matrix(w,sigma=0.1,epsilon=0.5):  # w:(n,n)
    n = w.shape[0]
    w = w /10000
    w[w==0] = np.inf
    w2 = w * w 
    w_mask = (np.ones([n,n]) - np.eye(n))
    return np.exp(-w2 / sigma) * (np.exp(-w2 / sigma) >= epsilon) * w_mask

def scaled_laplacian(a):
    "l_{scaled} = 2/ lambda * d^{-1/2} (d-a) d^{-1/2}-I"
    n = a.shape[0]
    d = np.sum(a,axis=1)
    l = np.diag(d) - a
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] >0:
                l[i,j] /= np.sqrt(d[i] * d[j])
    # lam = np.linalg.eigvals(l).max().real
    lam = np.linalg.eigvals(l).max()
    return 2 * l / lam - np.eye(n)


# 公式: t_k = 2l*t_{k-1}-t_{k-2}
def cheb_poly(l,ks):  # n,n  ks>=2
    n = l.shape[0]
    ll = [np.eye(n),l]
    for i in range(2,ks):
        ll.append(np.matmul(2*l,ll[-1]) - ll[-2])
    return np.asarray(ll)

if __name__ == "__main__":
    x = torch.randn(64,50,12,2)

    mat = np.random.randn(50,50)
    w = weight_matrix(mat)
    l = scaled_laplacian(w)
    lk = cheb_poly(l,ks=3)
    lk = torch.Tensor(lk.astype(np.float32))
    print(lk.shape)

    ks,kt = 3,3  # kernel_size of spatial and temporal
    bs = [[2,16,64],[64,16,64]]   # block size


    model = stgcn(ks=ks,kt=kt,bs=bs,lk=lk,t=12)
    y = model(x)
    print(x.shape,y.shape)
