from pathlib import Path
import os
import torch
from torch._C import dtype
from torch.nn.modules.loss import MSELoss
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from icecream import ic
# Téléchargement des données
from datamaestro import prepare_dataset
# import tensorflow as tf
import tensorboard as tb

#necessary in order to use add_embedding when both pytorch and tensorflow are installed in a same virtualenv
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)
savepath = Path("./model.pch")
BATCH_SIZE = 32
ITERATIONS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNIST_Dataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__() #Dit qu'on prend l'init de la classe parent
        self.data = data/255
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def __len__(self):
        return len(self.data)

train_data = DataLoader(MNIST_Dataset(train_images, train_labels), shuffle=True, batch_size=32)
test_data = DataLoader(MNIST_Dataset(test_images, test_labels), shuffle=True, batch_size=32)

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

    def forward(self, x):
        x = self.L1(x)

        x = self.relu(x)
        x = self.L3(x)

        x = self.relu(x)
        self.embedding = x.view(x.shape[0], x.shape[1]*x.shape[2])
        self.L2.weight = torch.nn.Parameter(self.L1.weight.T)
        self.L4.weight = torch.nn.Parameter(self.L3.weight.T)
        x = self.L4(x)
        x = self.relu(x)
        x = self.L2(x)
      
        x = self.sigmoid(x)

        return x

class State:
    def __init__(self, model, optim, criterion) -> None:
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.epoch, self.iteration = 0, 0

if savepath.is_file():
    with savepath.open('rb') as fp:
        state = torch.load(fp)
else:
    model = AE(len(train_data.dataset[0][0])).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=10e-5)
    criterion = MSELoss()
    state = State(model, optim, criterion)

for epoch in range(state.epoch, ITERATIONS):
    for x,y in train_data:
        state.optim.zero_grad()
        x = x.type(torch.float).to(device)

        y=y.type(torch.float)
        
        xhat = state.model(x)
        loss = state.criterion(xhat,x)
        loss.backward()
        state.optim.step()
        state.iteration += 1
        writer.add_pr_curve('pr_curve', x, xhat)
    ic(epoch, loss.item())
    writer.add_scalar('Loss/train', loss, epoch)
    with savepath.open('wb') as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)
writer.add_embedding(state.model.embedding)

#Images correspondant au décodage de l'interpolation des représentations dans l'espace latent
x1, x2 = x[-1].unsqueeze(0), x[-2].unsqueeze(0)
z1 = state.model.forward(x1)
z2 = state.model.forward(x2)
for lbd in [0.1*i for i in range(1,10)]:
    embedding = lbd*z1+(1-lbd)*z2
    writer.add_image(f'embedding with lambda={lbd}', embedding, 0)