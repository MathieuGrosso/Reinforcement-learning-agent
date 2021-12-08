
#  TODO:  Question 3 : Prédiction de séries temporelles
from utils import device,SampleMetroDataset,ForecastMetroDataset
from model import RNN
import torch
from icecream import ic
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn


# train_dataset = ForecastMetroDataset(train, stations_max=None)
# test_dataset  = ForecastMetroDataset(test, stations_max=train_dataset.stations_max)


# batch_size= 100


# train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle= True )
# test_dataloader  = DataLoader(test_dataset,batch_size,shuffle=True)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

# Nombre de stations utilisé
CLASSES = 80
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/AMAL-student_tp4.2021/data/"

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
train_data = (ForecastMetroDataset(matrix_train[:,:,:CLASSES,:DIM_INPUT],length=LENGTH))
train_dataloader =DataLoader((train_data),batch_size=BATCH_SIZE,shuffle=False)
test_dataloader = DataLoader(ForecastMetroDataset(matrix_test[:,:,:CLASSES,:DIM_INPUT],length=LENGTH,stations_max=train_data.stations_max),batch_size=BATCH_SIZE,shuffle=False)



train_features, train_labels = next(iter(train_dataloader))
ic(train_features.shape)
ic(train_labels.shape)

                  



#training and instanciate model 

criterion = torch.nn.MSELoss()
model     = RNN(in_features=160,hidden_size=100,out_features=160,criterion=criterion,opt='SGD',model='many to many',mode='forecast',ckpt_save_path='./ckpt/Trump/forecast80stations') #output=10 car il y a 10 stations pour l'isntant
model.fit(train_dataloader,validation_data=test_dataloader,n_epochs=100,lr=0.001)

# train, test = torch.load(PATH)

# train_dataset = ForecastMetroDataset(train, stations_max=None)
# test_dataset  = ForecastMetroDataset(test, stations_max=train_dataset.stations_max)

# batch_size = 1000
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=False)
# test_dataloader  = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, drop_last=False)

# model = RNN(in_features=160, out_features=160, hidden_size=100, opt='SGD', criterion='mse', model='many to many',mode='forecast',ckpt_save_path='./ckpt/Trump/forecast80stations')
# model.fit(train_dataloader,validation_data=test_dataloader,n_epochs=601, lr=0.005)
# print('\nDone')
