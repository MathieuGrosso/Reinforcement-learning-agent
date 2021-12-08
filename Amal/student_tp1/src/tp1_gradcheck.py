import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))
print(torch.autograd.gradcheck(mse, (yhat, y)))

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)
X = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5,6, requires_grad=True, dtype=torch.float64)
b = torch.randn(6,requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(linear, (X,W,b)))
