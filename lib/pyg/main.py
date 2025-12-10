import torch 
import torch.nn.functional as F
from gcn import dataset 
from gcn import model 

data = dataset[0]
print(data) # Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    pred = model(data.x, data.edge_index)
    loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
    print(f"================== epoch:{epoch+1:3d}/{200},Loss={loss:.2f} ==================")
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()