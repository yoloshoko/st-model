import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Transformer(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hid_dim=32, num_heads=4, layers=2, mlp_ratio=4, dropout=0,his_num=12,pred_num=6):
        super().__init__()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=hid_dim,
                                    kernel_size=(1, 1))
        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, dim_feedforward=hid_dim*mlp_ratio, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=layers)

        self.linear1 = nn.Linear(hid_dim,out_dim)
        self.linear2 = nn.Linear(his_num,pred_num)
    def forward(self, x):
        x = x.permute(0,3,2,1) # B,N,T,C -> B,C,T,N
        x = self.start_conv(x)  # B,C,T,N -> B,D,T,N
        x = x.permute(0,3,2,1)  # B,D,T,N -> B,N,T,D
        B,N,T,D = x.shape
        x = x.reshape(T,B*N,D)  # B,N,T,D -> T,B*N,D

        x = self.transformer_encoder(x)  # T,B*N,D -> T,B*N,D

        x = x.reshape(B,N,T,D)  # T,B*N,D -> B,N,T,D
        x = self.linear1(x) # B,N,T,D -> B,N,T,1
        x = x.squeeze(-1)  # B,N,T,1 -> B,N,T
        y = self.linear2(x)
        return y

if __name__ == "__main__":
    x = torch.randn(64,50,12,2) 
    model = Transformer()
    y = model(x)
    print(x.shape,y.shape)