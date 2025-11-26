import numpy as np 
import torch 
from torch import nn 
from torch.nn import functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        # 阶数（order）× 支撑结构数量（support_len）
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        # print(x.shape)
        out = [x]
        for a in support:
            x1 = self.nconv(x,a) # b,d,n,t
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a) # b,d,n,t
                out.append(x2) 
                x1 = x2

        h = torch.cat(out,dim=1) # b,nd,n,t
        h = self.mlp(h) # b,nd,n,t -> b,d,n,t 
        h = F.dropout(h, self.dropout, training=self.training)
        return h      # b,d,n,t

class gwn(nn.Module):
    def __init__(self, num_nodes=50, device="cpu",dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, 
                 aptinit=None, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwn, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # 统一使用Conv2d
                self.filter_convs.append(nn.Conv2d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size),
                    dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=skip_channels,
                    kernel_size=(1, 1)))
                
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                
                if self.gcn_bool:
                    self.gconv.append(gcn(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1,1),
            bias=True)

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1,1),
            bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):  
        input=input.permute(0,3,2,1)  # b,t,n,c -> b,c,n,t
        in_len = input.size(3) 
        # print(self.receptive_field) # r=13
        if in_len < self.receptive_field: # t < r
            x = nn.functional.pad(input, (self.receptive_field-in_len, 0, 0, 0)) # b,c,n,r
        else:
            x = input  # b,c,n,t
            
        x = self.start_conv(x)  # b,c,n,r -> b,c_,n,r
        skip = 0

        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1) # n,10 @ 10,n = n,n
            new_supports = self.supports + [adp]  # n,n

        for i in range(self.blocks * self.layers):
            
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x # b,c_,n,r=13
            # 所有卷积操作现在都处理4D输入
            filter = self.filter_convs[i](residual) # b,c_,n,r -> b,c_1,n,r_ (13 -> 12)
            filter = torch.tanh(filter) # b,c_,n,r_ (13 -> 12)
            # print(filter.shape)
            gate = self.gate_convs[i](residual) # b,c_1,n,r -> b,c_1,n,r_ (13 -> 12)
            gate = torch.sigmoid(gate)
            x = filter * gate # b,c_1,n,r_ @ b,c_1,n,r_ -> b,c_1,n,r_

            s = self.skip_convs[i](x) # b,c_1,n,r_ -> b,c_2,n,r_  (r_ = 12/10/9/7/6/4/3/1) -> (-1 -2 eg. 13-1=12 12-2=10 )
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip # b,c_2,n,r_

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports) # b,c_1,n,r_ -> b,c_1,n,r_
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            # print(x.shape,residual.shape) # b,c_1,n,r_(12) + b,c_1,n,r(13)
            x = x + residual[:, :, :, -x.size(3):] # b,c_1,n,r_ 
            x = self.bn[i](x) # b,c_1,n,r_ 
            # print(x.shape)

        x = F.relu(skip) # b,c_2=256,n,r_last=1
        x = F.relu(self.end_conv_1(x)) # b,c_3=512,n,r_last=1
        x = self.end_conv_2(x) # b,c_4=12,n,r_last=1
        x = x.squeeze(-1) # b,c_4,n
        return x

# def testPad():
#     x = torch.ones(1,12,1,2) # b,t,n,d
#     x = x.permute(0,3,2,1) # b,t,n,d -> b,d,n,t
#     print(x)
#     in_len = x.size(3)
#     receptive_field = 20
#     if in_len < receptive_field: # t < receptive_field
#         x = nn.functional.pad(x, (receptive_field-in_len, 0, 0, 0))
#     else:
#         x = x
#     print(x.shape)
#     print(x)

if __name__ == "__main__":
    x = torch.randn(64,12,50,2)
    mat = torch.randn(50,50)
    supports = [mat,mat.t()]
    model = gwn(supports=supports)
    y = model(x)
    print(x.shape,y.shape)