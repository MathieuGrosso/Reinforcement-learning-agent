import torch
from torch import nn
from torch.autograd import gradcheck
from torch.nn.modules.loss import MSELoss
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from icecream import ic
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

#Normalization
datax = (datax-torch.mean(datax))/torch.std(datax)
datay = (datay-torch.mean(datay))/torch.std(datay)

x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size = 0.1)

def linear(x,w,b):
    return x@w.T+b

def mse(yhat, y):
    return (1/y.shape[0])*torch.norm(yhat-y)**2

#Question 1

def train(x, y, batch_size=y_train.shape[0], epochs=100,lr=0.003):
    w = torch.randn(1, x.shape[1], requires_grad=True).to(torch.float)
    b = torch.randn(1,1, requires_grad=True).to(torch.float)

    for epoch in range(epochs):
        for batch in range(int(x.shape[0]/batch_size)):
            if batch_size*batch+batch <= x.shape[0]:
                x_batch = x[batch*batch_size:batch*batch_size+batch_size,]
                y_batch = y[batch*batch_size:batch*batch_size+batch_size,]
            else:
                x_batch = x[batch*batch_size:x.shape[0],]
                y_batch = y[batch*batch_size:y.shape[0],]
            loss = mse(linear(x_batch,w,b),y_batch)
            loss.backward()
            with torch.no_grad():
                w -= lr*w.grad
                b -= lr*b.grad
                w.grad.zero_()
                b.grad.zero_()
        if epoch%10==0:
            ic(epoch, loss)
        writer.add_scalar('Loss/train', loss, epoch) 
    return w, b, loss

def test(x,y):
    w, b, _ = train(x_train, y_train)
    return mse(linear(x,w,b),y)

ic(test(x_test,y_test))

ic(train(x_train, y_train)[-1])

#Question 2

def train_optim(x, y, epochs=1000,lr=0.001):
    w = torch.nn.Parameter(torch.randn(1, x.shape[1]).to(torch.float))
    b = torch.nn.Parameter(torch.randn(1).to(torch.float))
    optim = torch.optim.SGD(params=[w,b], lr=lr)
    optim.zero_grad()
    for epoch in range(epochs):
        x = nn.Linear(x.shape[1],w.shape[1])(x)
        x = nn.Tanh()(x)
        x = nn.Linear(x.shape[1],w.shape[1])(x)
        loss = nn.MSELoss()(x,y)
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()
        if epoch%100==0:
            ic(epoch, loss)
        writer.add_scalar('Loss/train', loss, epoch)
    return w, b, loss

# ic(train_optim(x_train,y_train, epochs=10)[-1])

def train_sequential(x, y, epochs=1000,lr=0.01):
    w = torch.nn.Parameter(torch.randn(1, x.shape[1]).to(torch.float))
    b = torch.nn.Parameter(torch.randn(1).to(torch.float))
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], w.shape[1]),
        nn.Tanh(),
        nn.Linear(x_train.shape[1], w.shape[1]),
    )
    for epoch in range(epochs):
        yhat = model(x)
        loss = nn.MSELoss()(yhat,y)
        if epoch%100 == 0:
            ic(epoch, loss.item())
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= lr*param.grad
    return loss.item()

ic(train_sequential(x_train, y_train))