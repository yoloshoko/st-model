import torch 
from torch import nn  

class mlp(nn.Module):
    def __init__(self,input_size=2,output_size=1,hiddens=[16,16],his_num=12,pred_num=6):
        super().__init__()
        self.fcs = nn.ModuleList()
        self.acs = nn.ModuleList()
        prev = input_size
        for hid in hiddens:
            self.fcs.append(nn.Linear(prev,hid))
            self.acs.append(nn.Sigmoid())
            prev = hid 
        
        self.linear1 = nn.Linear(hiddens[-1],output_size)
        self.linear2 = nn.Linear(his_num,pred_num)
    def forward(self,x):
        b,n,t,c = x.size()
        y = x
        # zip 会在较短的迭代器结束时停止
        for fc,ac in zip(self.fcs,self.acs): 
            y = ac(fc(y)) # b,n,t,c -> b,n,t,hiddens[-1]

        y = self.linear1(y)  # b,n,t,hiddens[-1] -> b,n,t,1
        y = y.squeeze(-1)
        y = self.linear2(y)
        return y


if __name__ == "__main__":
    # b,n,t,c
    x = torch.randn(64,50,12,2)
    model = mlp()
    y = model(x)

    print(x.shape,y.shape)