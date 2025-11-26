import torch 
from torch import nn 
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.nn import GATConv
from torch.nn import functional as F 

class metaLearner(nn.Module):
    def __init__(self,in_dim=2,out_dim=1,pred_num=12,his_num=6,d_mk=16):
        super().__init__()
        self.gru = nn.GRU(in_dim,out_dim)
        self.gat = GATConv(pred_num * in_dim,pred_num)

        self.gamma = nn.Parameter(torch.FloatTensor(pred_num))
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(pred_num,d_mk)
    def forward(self,x,edge_index):
        b,n,t,c = x.size()
        x_t = x
        x_gru = x.reshape(b*n,t,c)
        x_gru,_ = self.gru(x_gru) # bn,t,c -> bn,t,1 -> bn,t
        x_gru = x_gru.squeeze(-1)

        x_gat = x.reshape(b*n,t*c)
        x_gat = self.gat(x_gat,edge_index) # bn,tc -> bn,t
        

        x_mk = self.sigmoid(self.gamma) * x_gat + self.sigmoid(1 - self.gamma) * x_gru # bn,t * t -> bn,t

        x_mk = self.linear(x_mk) # bn,t -> bn,d_mk
        x_mk = x_mk.reshape(b,n,-1) # b,n,d_mk

        return x_mk

class metaLinear(nn.Module):
    def __init__(self,in_1,in_2,out):
        super().__init__()
        self.in_1 = in_1
        self.in_2 = in_2 
        self.out = out
        self.w_linear = nn.Linear(in_1,in_2 * out)
    def forward(self,x1,x2):
        f"b,n,in_1  b,n,in_2"
        b,n,in_1 = x1.size()
        b,n,in_2 = x2.size()
        x1 = x1.reshape(b*n,in_1) # bn,in_1 -> b,n,in_1
        w = self.w_linear(x1) # b,n,in_1 -> b,n,in_2*out
        w = w.reshape(b*n,in_2,self.out) # b,n,in_2*out -> b*n,in_2,out
        x2 = x2.reshape(b*n,1,in_2) # b,n,in_2 -> b*n,1,in_2
        y = torch.bmm(x2,w) # b*n,1,in_2 @ b*n,in_2,out = b*n,1,out
        y = y.squeeze(dim=1).reshape(b,n,-1) # b*n,1,out -> b*n,out -> b,n,out
        return y

class gruCell(nn.Module):
    def __init__(self,in_size,d_mk,hidden):
        super().__init__()
        self.r_x_linear = metaLinear(d_mk,in_size,hidden)
        self.r_h_linear = metaLinear(d_mk,hidden,hidden)
        self.z_x_linear = metaLinear(d_mk,in_size,hidden)
        self.z_h_linear = metaLinear(d_mk,hidden,hidden)
        self.c_x_linear = metaLinear(d_mk,in_size,hidden)
        self.c_h_linear = metaLinear(d_mk,hidden,hidden)
    def forward(self,x_i,x_mk,hidden):
        f"b,n,c  b,n,d_mk   b,n,d"
        r = F.sigmoid( self.r_x_linear(x_mk,x_i)  + self.r_h_linear(x_mk,hidden) )  # b,n,d
        z = F.sigmoid( self.z_x_linear(x_mk,x_i)  + self.z_h_linear(x_mk,hidden) )  # b,n,d
        c = F.tanh( self.c_x_linear(x_mk,x_i)  + r * self.c_h_linear(x_mk,hidden) ) # b,n,d
        next_hidden = (1 - z) * hidden + z * c # b,n,d
        return next_hidden

class stNet(nn.Module):
    def __init__(self,hidden=16,d_mk=16,in_size=2,out_size=1):
        super().__init__()
        self.hidden = hidden
        self.d_mk = d_mk
        self.in_size = in_size
        self.gc = gruCell(in_size,d_mk,hidden) 
        self.fc = nn.Linear(hidden,out_size)
    def forward(self,x,x_mk):
        b,n,t,c = x.size()
        hidden = torch.zeros(b,n,self.hidden) # b,n,d
        h_outputs = []
        for i in range(x.shape[2]):
            x_i = x[:,:,i,:] # b,n,c
            f"b,n,c  b,n,d_mk   b,n,d"
            hidden = self.gc(x_i,x_mk,hidden) # b,n,d
            out = self.fc(hidden) # b,n,d -> b,n,1
            out = out.squeeze(-1) # b,n,1 -> b,n
            h_outputs.append(out) # t *  [b,n]
        output = torch.stack(h_outputs,dim=0) # t,b,n
        output = output.permute(1,2,0) # b,n,t
        # print(output.shape)
        return output

class metaStnn(nn.Module):
    def __init__(self,his_num=12,pred_num=6):
        super().__init__()
        self.ml = metaLearner() 
        self.stn = stNet()
        self.ln = nn.Linear(his_num,pred_num)
    def forward(self,x,edge_index): 
        x_mk = self.ml(x,edge_index) # b,n,d_mk
        # print(x_mk.shape)
        learned_adj = torch.bmm(x_mk,x_mk.permute(0,2,1)) # b,n,k * b,k,n -> b,n,n
        learned_adj = F.relu(learned_adj)
        out = self.stn(x,x_mk)
        out = F.relu(out)
        out = self.ln(out)
        return out,learned_adj
    

if __name__ =="__main__":
    # b,n,t,c
    x = torch.randn(64,50,12,2)
    edge_index = erdos_renyi_graph(50,0.5)

    # print(torch.FloatTensor(6))
    model = metaStnn()
    out,learned_adj = model(x,edge_index)
    print(x.shape,out.shape,learned_adj.shape)