import torch 
from torch import nn
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.nn import GATConv
from torch.nn import functional as F

class gat(nn.Module):
    def __init__(self,num_layers=4,input_size=2,output_size=1,num_heads=3,his_num=12,pred_num=6):
        super().__init__()
        self.convs = nn.ModuleList()
        in_size = input_size * his_num
        out_size = output_size * his_num
        self.convs.append(
            GATConv(in_size,out_size,heads=num_heads,concat=True)
        )
        self.num_layer = num_layers
        for _ in range(num_layers-2):
            self.convs.append(
                GATConv(out_size * num_heads, out_size,heads=num_heads,concat=True)
            )
        self.convs.append(
                GATConv(out_size * num_heads, out_size,heads=num_heads,concat=False)
        )
        
        self.linear1 = nn.Linear(out_size, his_num)
        self.linear2 = nn.Linear(his_num,pred_num)
    def forward(self,x,edge_index):
        b,n,t,c = x.size()
        y = x.reshape(b*n,t*c)
        for i in range(self.num_layer):
            y = self.convs[i](y,edge_index) # b*n,t*c_in -> b*n,t*c_out*h -> b*n,t*c_out
            if i != self.num_layer -1 :
                y = F.relu(y)
        y = self.linear1(y) # b*n,t*c_out -> b*n,t
        y = self.linear2(y) # b*n,t -> b*n,p
        y = y.reshape(b,n,-1)
        return y

if __name__ == "__main__":
    # b,n,t,c
    x = torch.randn(64,50,12,2) 
    # Returns the edge_index of a random Erdos-Renyi graph.
    edge_index = erdos_renyi_graph(num_nodes=50, edge_prob=0.5) 
    print(edge_index)
    model = gat()
    y = model(x,edge_index)
    print(x.shape,y.shape)
