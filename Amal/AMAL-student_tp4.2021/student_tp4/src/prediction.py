from utils import device,SampleMetroDataset
from model import RNN
import torch
from icecream import ic
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn



#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
train, test = torch.load('/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/AMAL-student_tp4.2021/data/hzdataset.pch')


train = train[:, :, :10, :]
test = test[:, :, :10, :]


train_dataset = SampleMetroDataset(train, stations_max=None)
test_dataset  = SampleMetroDataset(test, stations_max=train_dataset.stations_max)


batch_size= 1000


train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle= True )
test_dataloader  = DataLoader(test_dataset,batch_size,shuffle=True)





#Understanding data: 
train_features, train_labels = next(iter(train_dataloader)) #train feature shape: batch, lenght, dim
# train_features = train_features.view(train_features.shape[1],train_features.shape[0],train_features.shape[2])
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
# print(f"Label: {train_labels.shape}")



#training and instanciate model 
criterion = torch.nn.CrossEntropyLoss()
model     = RNN(in_features=1,hidden_size=30,out_features=10,mode='classification',model="one to many",criterion=criterion,opt='SGD',ckpt_save_path='./ckpt/classification/hidden20') #output=10 car il y a 10 stations pour l'isntant
model.fit(train_dataloader,validation_data=test_dataloader, n_epochs=200, lr=0.0001, verbose=1)


