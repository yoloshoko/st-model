import torch 
from torch import nn 
from torch.nn import functional as F

class linear(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.mlp = nn.Conv2d(cin,cout,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=True)

    def forward(self,x):
        out = self.mlp(x)
        return out

class nconv(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,a):
        out = torch.einsum("bcnt,nm->bcmt",(x,a))
        return out


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout=0.1,support_len=3,order=2):
        super().__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) *c_in 
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
    def forward(self,x,supports):
        out = [x]
        for a in supports:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2,self.order+1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h,self.dropout)
        return h

class gwn(nn.Module):
    def __init__(self,supports=None,blocks=4,layers=2,kernel_size=2,residual_channels=32,dilation_channels=32,skip_channels=256,in_dim=2,pred_num=6,end_channels=512):
        super().__init__()
        self.supports = supports
        self.start_conv = nn.Conv2d(in_dim,residual_channels,kernel_size=(1,1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.receptive_field = 1

        self.nodevec1 = nn.Parameter(torch.randn(50, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, 50), requires_grad=True)

        self.blocks = blocks
        self.layers = layers
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(
                    residual_channels,
                    dilation_channels,
                    kernel_size=(1,kernel_size),
                    dilation=new_dilation
                ))
                self.gate_convs.append(nn.Conv2d(
                    residual_channels,
                    dilation_channels,
                    kernel_size=(1,kernel_size),
                    dilation=new_dilation
                ))
                self.residual_convs.append(nn.Conv2d(
                    dilation_channels,
                    residual_channels,
                    kernel_size=(1,1)
                ))
                self.skip_convs.append(nn.Conv2d(
                    dilation_channels,
                    skip_channels,
                    kernel_size=(1,1)
                ))
                new_dilation *=2
                self.receptive_field +=additional_scope
                additional_scope *=2

                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(gcn(
                    dilation_channels,
                    residual_channels
                ))
        
        self.end_conv_1 = nn.Conv2d(skip_channels,end_channels,kernel_size=(1,1))
        self.end_conv_2 = nn.Conv2d(end_channels,pred_num,kernel_size=(1,1))
    def forward(self,x):
        b,n,t,c = x.size()
        x = x.reshape(b,c,n,t) # b,c,n,t

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x =F.pad(x, (self.receptive_field - in_len,0))  # b,c,n,t
        x = self.start_conv(x) # b,c,n,t -> # b,c_r,n,t
        skip = 0
        
        adp = F.softmax(F.relu(torch.mm(self.nodevec1,self.nodevec2)),dim=1) # n,10 @ 10,n = n,n
        new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x 
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)

            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)

            x = filter * gate 

            s = self.skip_convs[i](x)
            try:
                skip = skip[:,:,:,-s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = self.gconv[i](x,new_supports)
            x = x + residual[:,:,:,-x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.squeeze(-1)
        x = x.permute(0,2,1)
        return x
    

if __name__ == "__main__":
    # b,n,t,c
    x = torch.randn(64,50,12,2)
    mat = torch.randn(50,50)
    supports = [mat, mat.t()]
    model = gwn(supports=supports)
    y = model(x)
    print(x.shape,y.shape)