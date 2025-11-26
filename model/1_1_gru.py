import torch
from torch import nn 

class gru(nn.Module):
    def __init__(self,input_size=2,hidden_size=16,output_size=1,his_num=12,pred_num=6):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,hidden_size=hidden_size, \
                          num_layers=1,batch_first=True)
        self.linear1 = nn.Linear(hidden_size,output_size)
        self.linear2 = nn.Linear(his_num,pred_num)
    def forward(self,x): 
        assert x.dim() == 4
        b,n,t,c = x.size()
        x = x.reshape(b*n,t,c)
        y,_ = self.gru(x) # b*n,t,c -> b*n,t,h and 1,b*n,h
        y = self.linear1(y) # b*n,t,h -> b*n,t,1
        y = y.squeeze(-1).reshape(b,n,t) # b*n,t,1 -> b*n,t -> b,n,t
        y = self.linear2(y)

        return y
    
if __name__ == "__main__":
    # b,n,t,c
    x = torch.randn(64,50,12,2)
    model = gru()
    y = model(x)

    print(x.shape,y.shape)