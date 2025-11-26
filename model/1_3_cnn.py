import torch 
from torch import nn 

class cnn(nn.Module):
    def __init__(self,input_size=2,output_size=1,his_num=12,pred_num=6):
        super().__init__()
        self.relu = nn.ReLU()
        in_channel = [2,16,16,32]
        out_channel = [16,16,32,32]
        kernel_sizes = [(16,3),(3,5),(8,3),(4,3)]
        pool_kernel_sizes = [(2,1)]

        self.conv1 = nn.Conv2d(in_channels=in_channel[0],out_channels=out_channel[0],kernel_size=kernel_sizes[0],padding='same')
        self.conv2 = nn.Conv2d(in_channels=in_channel[1],out_channels=out_channel[1],kernel_size=kernel_sizes[1],padding='same')
        self.conv3 = nn.Conv2d(in_channels=in_channel[2],out_channels=out_channel[2],kernel_size=kernel_sizes[2],padding='same')
        self.conv4 = nn.Conv2d(in_channels=in_channel[3],out_channels=out_channel[3],kernel_size=kernel_sizes[3],padding='same')

        self.pool = nn.AvgPool2d(kernel_size=pool_kernel_sizes[-1])
        kernel0,kernel1 = pool_kernel_sizes[-1][0],pool_kernel_sizes[-1][1]
        # print(kernel0,kernel1)
        # input_dim = int ( out_channel[-1]/pool_kernel_sizes[0] ) * int ( nodes / pool_kernel_size[1] )
        in_size = int(out_channel[-1] / kernel0  *   his_num /kernel1)
        self.linear = nn.Linear(in_size , pred_num)
    def forward(self,x):
        b,n,t,c = x.size()
        x = x.permute(0,3,2,1) # b,n,t,c -> b,c,t,n
        y = self.conv1(x) # b,c,t,n -> b,c1,t,n
        y = self.relu(y) 
        y = self.conv2(y) # b,c1,t,n -> b,c2,t,n
        y = self.relu(y)
        y = self.conv3(y) # b,c2,t,n -> b,c3,t,n
        y = self.relu(y)
        y = self.conv4(y) # b,c3,t,n -> b,c4,t,n
        y = self.relu(y)

        y = self.pool(y) # b,c4,t,n -> b,c4,[t/2],[n/1] -> b,c4,[t/2],n
        y = y.permute(0,3,1,2).reshape(b,n,-1) # b,c4,[t/2],n -> b,n,c4*[t/2]
        
        y = self.linear(y)
        return y
    
if __name__ == "__main__":
    # b,n,t,c 
    x = torch.randn(64,50,12,2)
    model = cnn()
    y = model(x)
    print(x.shape,y.shape)