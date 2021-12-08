from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from icecream import ic



import numpy as np
import datetime
# Téléchargement des données
from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");

import argparse
parser = argparse.ArgumentParser(prog='TP3')
parser.add_argument("--Path",help="allow to choose the mode of the agent", action="store")
parser.add_argument("--Model",help="allow to choose the mode of the agent", action="store")
parser.add_argument("--Mixed",action="store")
args=parser.parse_args()



train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels   =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


#pour utiliser un gpu : 

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)

#initialize
from pathlib import Path 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size=128
ITERATIONS=100

class MNIST_dataset(Dataset):
    def __init__(self, data,label, transform=None):
        self.data=data/255
        self.label=label

    def __getitem__(self,idx):
        return self.data[idx],self.label[idx]

    def __len__(self):
        return len(self.data)

train_data = DataLoader(MNIST_dataset(train_images,train_labels),shuffle=True,batch_size=batch_size)
test_data  = DataLoader(MNIST_dataset(test_images,test_labels),shuffle=True,batch_size=batch_size)


#print(image)
def print_image(data):

    figure= plt.figure (figsize=(12,8))
    cols, rows= 3,3 
    for i in range(1,cols*rows+ 1):
        img,label = next(iter(data))
        figure.add_subplot(rows,cols,i)
        plt.title(label[i])
        plt.axis("off")
        plt.imshow(img[i],cmap='gray')
    plt.show()

# print_image(train_data)
# print_image(test_data)


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear  = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear     = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate       = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f           = f

        self.flatten = nn.Flatten(start_dim=1)
        self.Linear = nn.Linear(784,10)

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate      = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear    = self.linear[layer](x)

            x         = gate * nonlinear + (1 - gate) * linear
        x=self.flatten(x)
        # ic(x.shape)
        x=self.Linear(x)
        # ic(x.shape)
            
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.relu  = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=2) # divise par deux
        self.conv2 = nn.Conv2d(6,9,3,2) # divise par 2
        self.batch_norm1 = nn.BatchNorm2d(9)
        self.conv3 = nn.Conv2d(9,10,3,2) #divise par 2
        self.batch_norm2 = nn.BatchNorm2d(10)
        self.out = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1)
        self.Linear = nn.Linear(40,10)


        

    def forward(self, x):
        x=x.view(x.shape[0],1,x.shape[1],x.shape[2])
        # ic(x.shape)
        out=self.conv1(x)
        # ic(out.shape)
        out=self.relu(out)
        
        out=self.conv2(out)
        # ic(out.shape)
        out=self.relu(out)
        out=self.batch_norm1(out)
        out=self.conv3(out)
        out=self.batch_norm2(out)
        out=self.flatten(out)
        out=self.Linear(out)
        out=self.out(out)
    
        # ic(out.shape)
        return out


class AE(nn.Module):
    def __init__(self, input_shape, hidden_shape=128, output_shape=32):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.L1 = nn.Linear(
            in_features=self.input_shape, out_features=self.hidden_shape
        )
        self.L2 = nn.Linear(
            self.hidden_shape, self.input_shape
        )
        self.L3 = nn.Linear(
            in_features = self.hidden_shape, out_features=self.output_shape
        )
        self.L4 = nn.Linear(
            in_features = self.output_shape, out_features = self.hidden_shape
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.embedding = []

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28*32,10)
        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self,x):
        # ic(x.shape)
        encoded = self.L1(x)
        
        encoded = self.relu(encoded)
        encoded = self.L3(encoded)
        encoded = self.relu(encoded)
        # ic(encoded.shape)

        encoded_out = self.classifier(encoded)


        self.L2.weight = torch.nn.Parameter(self.L1.weight.T)
        self.L4.weight = torch.nn.Parameter(self.L3.weight.T)

        decoded = self.L4(encoded)
        decoded = self.relu(decoded)
        decoded = self.L2(decoded)
        decoded = self.sigmoid(decoded)
        # ic(decoded.shape)
        

        return decoded,encoded_out


# class AE(nn.Module):
#     def __init__(self, input_shape, n_embedded):
#         super(AE,self).__init__()
#         self.input_shape = input_shape
        
#         self.encoder=nn.Sequential(torch.nn.Linear(self.input_shape, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 36),
#             torch.nn.ReLU(),
#             torch.nn.Linear(36, 18),
#             torch.nn.ReLU(),
#             torch.nn.Linear(18, n_embedded),
#             nn.ReLU())

 
        
#         # must return 128,10
#         self.classifier = nn.Sequential(
#             nn.Flatten(start_dim=1),
#             nn.Linear(280,10)
#         )
        
#         self.decoder = nn.Sequential(
#             torch.nn.Linear(n_embedded, 18),
#             torch.nn.ReLU(),
#             torch.nn.Linear(18, 36),
#             torch.nn.ReLU(),
#             torch.nn.Linear(36, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, self.input_shape),
#             torch.nn.Sigmoid()
#         )

#         self.softmax = nn.Softmax(dim=1)
        

#     def forward(self,x):
#         encoded = self.encoder(x)

#         # ic(encoded.shape)
#         encoded_out = self.classifier(encoded)
#         # ic(encoded_out.shape)
#         decoded = self.softmax(encoded)
#         decoded = self.decoder(encoded)
#         return decoded,encoded_out

class State : 
    def __init__(self,model,optim,criterion1,criterion2):
        self.model = model 
        self.optim = optim 
        self.epoch, self.iteration = 0,0 
        self.criterion1 = criterion1
        self.criterion2 = criterion2



#training 
""""to make training work: 
tp3.py --Model name_model --Path name_path_for_model_checkpoint 
--Mixed True_or_False 
(use true for AE, and false for the other models)
example: python3 tp3.py --Model Net  --Path .NET.pch --Mixed False.
example2: python3 tp3.py --Model Highway_Network  --Path .HN.pch --Mixed False
example3: python3 tp3.py --Model AE --path .AE.pch --Mixed True"""



savepath = Path(args.Path)
if savepath.is_file():
    with savepath.open('rb') as fp : 
        state  = torch.load(fp)
else: 
    
    if args.Model=='Net':
        print('model is simple Net')
        model = Net()
    if args.Model=='Highway_Network':
        print('model is highway')
        model = Highway(size=len(train_data.dataset[0][0]),num_layers=2,f=torch.tanh)
    if args.Model=='AE':
        print('model is AE')
        model = AE(input_shape=len(train_data.dataset[0][0])) #il y a 10 class
    optim      = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.CrossEntropyLoss()
    state      = State(model, optim,criterion1,criterion2) 

mse_multp = 0.5
cls_multp = 0.5
for epoch in range(state.epoch,ITERATIONS):
    total_mseloss = 0.0
    total_clsloss = 0.0
    for x,y in train_data:
        x = x.type(torch.float).to(device)
        y = y.type(torch.long).to(device)
        if args.Mixed=='True':
            decoded, encoded=state.model(x)
            loss_mse = criterion1(decoded,x)
            loss_cls = criterion2(encoded,y)
            loss     = (mse_multp*loss_mse) + (cls_multp*loss_cls)
        else: 
            yhat=state.model(x)
            loss_cls = criterion2(yhat,y)
            loss = loss_cls

        state.optim.zero_grad()    
        loss.backward()
        state.optim.step()
        state.iteration += 1
        if args.Mixed=='True':
            writer.add_pr_curve('AE/pr_curve',x,decoded)
            total_clsloss += loss_cls.item()
            total_mseloss += loss_mse.item()
            writer.add_scalar('/Loss/MSE/'+args.Model+'/train',loss_cls,epoch)
            writer.add_scalar('/Loss/CLS/'+args.Model+'/train',loss_mse,epoch)
    
        else: 
            total_clsloss += loss_cls.item()
            writer.add_scalar('/Loss/CLS/'+args.Model+'/train',loss_cls,epoch)
    if args.Mixed=='True':
        print('epoch [{}/{}], loss_mse: {:.4f}  loss_cls: {:.4f}  '.format(epoch+1, state.epoch, total_mseloss / len(train_data), total_clsloss / len(train_data)))
    else: 
        print('epoch [{}/{}],  loss_cls: {:.4f}  '.format(epoch+1, state.epoch,  total_clsloss / len(train_data)))
    with savepath.open("wb") as fp: 
        state.epoch=epoch+1
        torch.save(state,fp)

#Images correspondant au décodage de l'interpolation des représentations dans l'espace latent
x1, x2 = x[-1].unsqueeze(0), x[-2].unsqueeze(0)
z1 = state.model.forward(x1)
z2 = state.model.forward(x2)
for lbd in [0.1*i for i in range(1,10)]:
    embedding = lbd*z1+(1-lbd)*z2
    writer.add_image(f'embedding with lambda={lbd}', embedding, 0)

writer.close()



