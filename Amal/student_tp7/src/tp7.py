import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split,Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from torch import optim
import numpy as np 
import datetime
from icecream import ic
from utils import *
from datamaestro import prepare_dataset




# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
BATCH_SIZE  = 32


#  TODO:  Implémenter

## data : 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # #transform data: 
# data_transforms1 =   transforms.Compose(
#         [transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=45),
#          transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor()])
# data_transforms2 = transforms.Compose([transforms.ToPILImage(),transforms.RandomRotation(degrees=45),transforms.RandomResizedCrop(size=(18, 18)),transforms.ToTensor()])

# data loading
logging.info("Loading datasets...")
ds = prepare_dataset("com.lecun.mnist")




train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
train_dataset = MNIST(train_images, train_labels, device=device)
test_dataset  = MNIST(test_images, test_labels,device = device)
new_len_train = int(TRAIN_RATIO * len(train_dataset))
new_len_test  = int(TRAIN_RATIO * len(test_dataset))


train_dataset, val_dataset = random_split(dataset=train_dataset,
                                          lengths=[int(new_len_train), int(len(train_dataset) - new_len_train)],
                                          generator=torch.Generator().manual_seed(42069))


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)




input,label=next(iter(train_dataloader))

writer = SummaryWriter()
    


## training : 

class training_global(nn.Module): 
    def __init__(self,criterion,device,opt,weight_decay,regularization=None,ckpt_save_path=None):
        super().__init__()
        self.criterion = criterion
        self.device = device
        self.regularization = regularization
        self.optimizer = opt
        self.state={}
        self.ckpt_save_path= ckpt_save_path
        self.weight_decay = weight_decay
    
    def __train_test_epoch(self,traindata,testdata,epoch):
        epoch_train_loss = 0
        epoch_train_acc  = 0 
        epoch_test_loss  = 0
        epoch_test_acc   = 0 
        for idx, data in enumerate(traindata):
            
            self.opt.zero_grad()
            input,label = data
            
            input = input.reshape(-1,self.input_size)
            b_size,n_features = input.shape
            if b_size==BATCH_SIZE:
                output = self(input,'Train',epoch) # self c'est le modèle
                loss = self.criterion(output,label)
                if self.regularization == 'L1':
                    l1_lambda = 0.007253
                    l1_penalty = sum(p.abs().sum() for p in self.parameters())
                    loss = loss + l1_lambda * l1_penalty
                if self.regularization =="L2":
                    l2_lambda = 0.07253
                    l2_penalty = sum(p.pow(2.0).sum() for p in self.parameters())
                    loss = loss + l2_lambda*l2_penalty

                n_correct = (torch.argmax(output,dim=1)==label).sum().item()
                total     = label.size(0)
            
                epoch_train_acc += n_correct/total
                loss.backward()
                self.opt.step()
                epoch_train_loss += loss.item()

        with torch.no_grad():
            for idx,data in enumerate(testdata):
                input,label = data
                input = input.reshape(-1,self.input_size)
                b_size,n_features = input.shape
                if b_size==BATCH_SIZE:
                    output = self(input,'Test',epoch) # self c'est le modèle
                    loss = self.criterion(output,label)
                    n_correct = (torch.argmax(output,dim=1)==label).sum().item()
                    total     = label.size(0)
                    epoch_test_acc += n_correct/total
                    epoch_test_loss += loss.item()
        


        return epoch_train_loss/len(traindata),epoch_train_acc/len(traindata),epoch_test_loss/len(testdata),epoch_test_acc/len(testdata)

    def __weights_histo(self,epoch):
        for name, param in self.named_parameters(): 
            writer.add_histogram(name, param, global_step=epoch, bins='tensorflow')

    def __validate(self, dataloader,epoch):
        epoch_loss = 0
        epoch_acc  = 0
        for idx, data in enumerate(dataloader):
            self.opt.zero_grad()
            input,label = data
            
            input = input.reshape(-1,self.input_size)
            b_size,n_features = input.shape
            if b_size==BATCH_SIZE:
                output = self(input,'val',epoch) # self c'est le modèle
                loss = self.criterion(output,label)
                n_correct = (torch.argmax(output,dim=1)==label).sum().item()
                total     = label.size(0)
                epoch_acc += n_correct/total
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train_idx',loss,idx)
            
        return epoch_loss/len(dataloader), epoch_acc/len(dataloader)


    def fit(self, traindata,testdata,validation_data=None,batch_size=300, start_epoch=0, n_epochs=1000, lr=0.001, verbose=10,ckpt=None):
        ic(lr)
        
        parameters = self.parameters()
        if self.optimizer=="SGD":
            self.opt = optim.SGD(parameters,lr=lr,momentum=0.9)
        if self.optimizer=='Adam':
            self.opt = torch.optim.Adam(parameters, lr=lr, weight_decay=self.weight_decay, amsgrad=False)
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
      

        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
                g['lr'] = state['lr']
            
        for epoch in range(start_epoch,n_epochs):
            epoch_train_loss,epoch_train_acc,epoch_test_loss,epoch_test_acc = self.__train_test_epoch(traindata,testdata,epoch)
            print(f'\n Epoch {epoch+1} \n',
                        f'Train Loss= {epoch_train_loss:.4f}\n',f'Train Acc={epoch_train_acc:.4f}\n',f'test Loss= {epoch_test_loss:.4f}\n',f'Test Acc={epoch_test_acc:.4f}\n')
            
            
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/test',epoch_test_loss,epoch)
            writer.add_scalar('Acc/train',epoch_train_acc,epoch)
            writer.add_scalar('Acc/test',epoch_test_acc,epoch)
            if epoch % 10 ==0 : 
                self.__weights_histo(epoch) #using weight histograms to have the weights of each linear layer. 
            if validation_data is not None:
                with torch.no_grad():
                    val_loss, val_acc = self.__validate(validation_data,epoch)
                print('Epoch {:2d} loss_val: {:1.4f}  val_acc: {:1.4f} '.format(epoch+1, val_loss, val_acc))
                writer.add_scalar('Loss/val',val_loss,epoch)
                writer.add_scalar("Acc/val",val_acc,epoch)


            if self.ckpt_save_path:
                self.state['lr'] = lr
                self.state['epoch'] = epoch
                self.state['state_dict'] = self.state_dict()
                if not os.path.exists(self.ckpt_save_path):
                    os.mkdir(self.ckpt_save_path)
                torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{start_time}_epoch{epoch}.ckpt'))

   


class Net(training_global):
    def __init__(self,input_size,hidden_dim,output_size,criterion,device,opt,dropout, regularization=None,ckpt_save_path=None,weight_decay=0):
        super(Net,self).__init__(regularization = regularization,criterion =criterion,device = device,opt =  opt,ckpt_save_path= ckpt_save_path,weight_decay=weight_decay)
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim,output_size)
        self.activation = nn.ReLU()
        self.grads={}
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        

    def forward(self,x,mode,epoch):
        """model changed to add dropout, regularisation and batchnorm"""
        x1 = self.linear1(x)
        # x1 = self.dropout1(x1)
        x2 = self.linear2(x1)
        # x2 = self.dropout2(x2)
        out   = self.activation(self.linear3(x2))

        
        if mode == 'Train' and epoch%10==0 :
            # x.register_hook(self.save_grad("x"))
            x1.register_hook(self.save_grad("x1"))
            x2.register_hook(self.save_grad("x2"))
            out.register_hook(self.save_grad("out"))
            for key, value in self.grads.items():
                writer.add_histogram(key, value,  bins='tensorflow')

        return out 
    
    def save_grad(self,name):
        def hook(grad):
            self.grads[name] = grad
        return hook
    
    


criterion = torch.nn.CrossEntropyLoss()
regularization = "L1"
net = Net(input_size=test_dataset.n_features,hidden_dim=100,output_size=10,criterion=criterion,device=device,regularization = None,dropout=0.3, opt='Adam',weight_decay = 0.009348818325128058)
# net = Net(input_size=test_dataset.n_features,hidden_dim=100,output_size=10,criterion=criterion,device=device,regularization = None,dropout=0.0, opt='Adam',weight_decay = 0)
net.fit(train_dataloader,test_dataloader,val_dataloader,lr=0.0001)
        

# def forwardstore(x):
#     




