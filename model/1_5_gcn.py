import torch 
from torch import nn 

class gcn(nn.Module):
    def __init__(self,in_dim=2,out_dim=1,his_num=12,pred_num=6):
        super().__init__()
        self.cAtt = nn.Linear(in_dim,out_dim)
        self.tAtt = nn.Linear(his_num,pred_num)
    def forward(self,x,a):
        b,n,t,c = x.size()
        x = x.reshape(b,c,n,t)  # b,n,t,c -> b,c,n,t
        # 因为aij表示i节点对j节点的影响,\sum_i{xij aij} for any j
        out = torch.einsum('bcnt,nm->bcmt',x,a) 
        # out = torch.matmul(x,a) # b,c,n,t 错误:不满足算法的需求,但对于无向图的结果是一样的
        out = out.reshape(b,n,t,c) # b,c,n,t -> b,n,t,c
        out = self.cAtt(out).squeeze(-1) # b,n,t,c -> b,n,t,1 -> b,n,t
        out = self.tAtt(out) # b,n,t -> b,n,p
        return out

if __name__ == "__main__":
    # b,n,t,c
    x = torch.randn(64,50,12,2)
    a = torch.randn(50,50)
    model = gcn()
    y = model(x,a)
    print(x.shape,a.shape,y.shape)
