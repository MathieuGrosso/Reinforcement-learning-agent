
import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from os import path, makedirs
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import gradcheck
from icecream import ic 
from torch import optim
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser(prog='Qlearning, Sarsa parser')
parser.add_argument("--Model",help="allow to choose the mode of the agent", action="store")
args=parser.parse_args()

experiment_name = 'tensorboard-tp2'
exp_dir = './sample_projects/' + experiment_name

if not path.exists(exp_dir):
    makedirs(exp_dir)
writer = SummaryWriter('./sample_projects/tensorboard-tp2')


"""do :tensorboard --logdir=./sample_projects in terminal to print the results"""

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

#normalisation: 
datax = (datax-torch.mean(datax))/torch.std(datax)
datay = (datay-torch.mean(datay))/torch.std(datay)

#train_test_split: 
X_train, X_test, y_train, y_test = train_test_split(datax,datay,test_size=0.2)




# QUESTION 1:

def Linear(x,w,b):
    return x@w.T+b

def MSE(yhat, y):
    return (1/y.shape[0])*torch.norm(yhat-y)**2

#batch gradient descent: 
def train_batch(x,y, batch_size=y_train.shape[0], epochs=100,lr=0.003):
    w = torch.randn(1, x.shape[1], requires_grad=True).to(torch.float)
    b = torch.randn(1,1, requires_grad=True).to(torch.float)

    for epoch in range(epochs):
        yhat  = Linear(x,w,b)
        loss  = MSE(yhat,y)
        loss.backward()
        with torch.no_grad():
            w -= lr*w.grad
            b -= lr*b.grad
            w.grad.zero_()
            b.grad.zero_()
        if epoch%10==0:
            ic(epoch, loss)
        writer.add_scalar('Loss/train/batch descent', loss, epoch) 
    return w, b, loss

# stochastic gradient descent: 
def train_SGD(X,Y, batch_size=y_train.shape[0], epochs=100,lr=0.003):
    w = torch.randn(1, X.shape[1], requires_grad=True).to(torch.float)
    b = torch.randn(1,1, requires_grad=True).to(torch.float)

    for epoch in range(epochs):
        for x,y in zip(X,Y):
            yhat  = Linear(x,w,b)
            loss  = MSE(yhat,y)
            loss.backward()
            with torch.no_grad():
                w -= lr*w.grad
                b -= lr*b.grad
                w.grad.zero_()
                b.grad.zero_()
        if epoch%10==0:
            ic(epoch, loss)
        writer.add_scalar('Loss/train/SGD descent', loss, epoch) 
    return w, b, loss

#mini batch gradient descent
def train_minibatch(x, y, batch_size=y_train.shape[0], epochs=100,lr=0.003):
    w = torch.randn(1, x.shape[1], requires_grad=True).to(torch.float)
    b = torch.randn(1,1, requires_grad=True).to(torch.float)
    for epoch in range(epochs):
        for i in range(int(x.shape[0]/batch_size)):
            start_i = i*batch_size
            end_i = start_i + batch_size
            xb    = x[start_i:end_i]
            yb    = y[start_i:end_i]
            yhat  = Linear(x,w,b)
            loss  = MSE(yhat,y)
            loss.backward()
            with torch.no_grad():
                w -= lr*w.grad
                b -= lr*b.grad
                w.grad.zero_()
                b.grad.zero_()
        if epoch%10==0:
            ic(epoch, loss)
        writer.add_scalar('Loss/train/minibatch descent', loss, epoch) 

        
    return w, b, loss


def test(x,y,w,b,epochs=100):
    """dans le test on actualise plus les poids et biais"""
    for epoch in range(epochs):
        with torch.no_grad():
            yhat=model(x)
            test_loss=MSE(yhat,y)
        if epoch%2==0:
            ic(epoch,test_loss)
            writer.add_scalar('Loss/test/1st question', test_loss, epoch)    
       
    return 


w_batch,b_batch,loss_batch=train_batch(X_train,y_train)
ic(loss_batch)
ic(test(X_test,y_test,w_batch,b_batch))


w_SGD,b_SGD,loss_SGD=train_SGD(X_train,y_train)
ic(loss_SGD)
ic(test(X_test,y_test,w_SGD,b_SGD))

w_minibatch,b_minibatch,loss_minibatch=train_minibatch(X_train,y_train)
ic(loss_minibatch)
ic(test(X_test,y_test,w_minibatch,b_minibatch))

# QUESTION 2: 

def train2(x, y, batch_size=y_train.shape[0], epochs=100,lr=0.003):
    w = torch.nn.Parameter(torch.randn(1, x.shape[1]).to(torch.float))
    b = torch.nn.Parameter(torch.randn(1).to(torch.float))
    optim = torch.optim.SGD(params=[w,b], lr=lr)
    optim.zero_grad()
    for epoch in range(epochs):
        for i in range(int(x.shape[0]/batch_size)):
            start_i = i*batch_size
            end_i = start_i + batch_size
            xb    = x[start_i:end_i]
            yb    = y[start_i:end_i]
            yhat  = Linear(x,w,b)
            loss  = MSE(yhat,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if epoch%10==0:
                ic(epoch, loss)
            writer.add_scalar('Loss/train/with optim', loss, epoch) 
    return w,b,loss


def test2(x,y,w,b,epochs=100):
    """dans le test on actualise plus les poids et biais"""
    for epoch in range(epochs):
        with torch.no_grad():
            yhat=Linear(x,w,b)
            test_loss=MSE(yhat,y)
        if epoch%2==0:
            ic(epoch,test_loss)
            writer.add_scalar('Loss/test/With optim', test_loss, epoch)    
       
    return 




w,b,loss=train2(X_train,y_train)
ic(loss)
ic(test2(X_test,y_test,w,b))  



# QUESTION 3 Première Partie: 
class LinReg(nn.Module):
    def __init__(self):
        super(LinReg,self).__init__()
        self.n_channels=X_train.shape[1]
        self.linear_1=nn.Linear(self.n_channels,10)
        self.tan1=nn.Tanh()
        self.linear_2=nn.Linear(10,13)
        self.Loss=nn.MSELoss()

    def forward(self,x):
        out1=self.linear_1(x)
        out2=self.tan1(out1)
        out3=self.linear_2(out2)
        return out3

# QUESTION 3 Deuxième Partie:  
class LinRegSeq(nn.Module):
    def __init__(self):
        super(LinRegSeq,self).__init__()
        self.n_channels=X_train.shape[1]
    
        self.main=nn.Sequential(nn.Linear(self.n_channels,10),nn.Tanh(),nn.Linear(10,13))

    def forward(self,x):
        out=self.main(x)
        return out 

def fit(x, y, model, opt, batch_size=y_train.shape[0], epochs=100,lr=0.003):
    for epoch in range(epochs):
        model.train() #for training
        for i in range(int(x.shape[0]/batch_size)):
            start_i = i*batch_size
            end_i = start_i + batch_size
            xb    = x[start_i:end_i]
            yb    = y[start_i:end_i]
            yhat  = model(x)
            loss  = nn.MSELoss()(yhat,y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if epoch%10==0:
                ic(epoch, loss)
            writer.add_scalar('Loss/train/NN version', loss, epoch) 
        
        model.eval() #for evaluation
        with torch.no_grad():
            yhat=model(x)
            test_loss=nn.MSELoss()(yhat,y)
            writer.add_scalar('Loss/test/NN version', loss, epoch) 
    return loss,test_loss
    

def get_model(model):
    if model=='LinReg':
        model=LinReg()
    if model =='LinRegSeq':
        model=LinRegSeq()
        
    return model, optim.SGD(model.parameters(),lr=0.003)


model, opt = get_model('LinRegSeq')
ic(fit(X_train,y_train,model,opt))


model, opt = get_model('LinReg')
ic(fit(X_train,y_train,model,opt))




writer.close()