import logging
logging.basicConfig(level=logging.INFO)
import datetime
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
from AE import * 

from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Ratio du jeu de train à utiliser

TRAIN_BATCHSIZE  = 32
TEST_BATCHSIZE   = 32
opt = 'Adam'

#  TODO:  Implémenter

## data : 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# data loading
logging.info("Loading datasets...")
ds = prepare_dataset("com.lecun.mnist")




train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
train_dataset = MNIST(train_images, train_labels, device=device)
test_dataset  = MNIST(test_images, test_labels,device = device)

train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCHSIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=TEST_BATCHSIZE)

input,label = next(iter(train_dataloader))

TRAIN_BATCHSIZE = 32
img_list = []
G_losses = []
D_losses = []
iters = 0

num_epochs = 100



class VAE(nn.Module): 
    def __init__(self,dim_latent,dim_hidden,dim_out,device,opt,weight_decay,regularization=None,ckpt_save_path=None,vae=None):
        super(VAE,self).__init__()
        self.device = device
        self.regularization = regularization
        self.optimizer = opt
        self.vae = vae
        self.state={}

        self.ckpt_save_path= ckpt_save_path
        self.weight_decay = weight_decay
        self.xdim = 28 * 28
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        if self.vae == "MLP":
            ic('we use linear encoder')
            self.encoder = EncoderLinear(dim_in=self.xdim,dim_hidden=self.dim_hidden,dim_out=self.dim_out)
            self.decoder = DecoderLinear(dim_in=self.dim_latent,dim_hidden=self.dim_hidden,dim_out=self.xdim)
            self.fc_mu = nn.Linear(self.dim_out,self.dim_latent)
            self.fc_var = nn.Linear(self.dim_out,self.dim_latent)
        if self.vae =='Conv':
            ic("we use convolutional encoder")
            self.encoder = EncoderConv(nchannels=1,hchannels=16)
            self.decoder = DecoderConv(nchannels=1,hchannels=16,dimin = self.dim_latent)
            self.fc_mu = nn.Linear(32*4*4,self.dim_latent)
            self.fc_var = nn.Linear(32*4*4,self.dim_latent)
        

    
    def reconstruction_loss(self,x_logits,x):
        #first compute proba : 
        # recon_loss = F.binary_cross_entropy_with_logits(x_logits,x).sum(-1).mean()
        recon_loss = F.mse_loss(torch.sigmoid(x_logits), x, reduction='none').sum(dim=1).mean()

        return recon_loss



    def __train_test_epoch(self,traindata,testdata,epoch):
        epoch_train_loss = 0
        epoch_test_loss  = 0
        for idx, data in enumerate(traindata):
            self.opt.zero_grad()
            x,label = data
            b_size = x.shape[0]
  
         
            x  =  x.view(-1, self.xdim)
            x_encoded = self.encoder(x)
           
            #get distributions
            mu  = self.fc_mu(x_encoded)
            logvar = self.fc_var(x_encoded)
            std=torch.sqrt(torch.exp(logvar))

            q_zx = torch.distributions.Normal(mu,std)
            p_z = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std)) #gaussienne
            
            
            z   = q_zx.rsample() #random vector from latent distributions
            
            #decode 
            x_logits = self.decoder(z)  # x hat is not an image but a parameter for the distribution. 
            x_hat = torch.sigmoid(x_logits)

            #reconstruction loss: 
            recon_loss = self.reconstruction_loss(x_logits,x) 
      
            #kl divergence: 
            log_qzx = q_zx.log_prob(z)
            log_pz = p_z.log_prob(z)
            kl = log_qzx-log_pz
            kl = kl.sum(dim=-1).mean()

            #elbo loss: 
            elbo = kl+recon_loss
            elbo.backward()
            self.opt.step()
            epoch_train_loss += elbo.item()
        


        with torch.no_grad():
            for idx, data in enumerate(testdata):
                x,label = data
                b_size = x.shape[0]
                x  =  x.view(-1, self.xdim)
                x_encoded = self.encoder(x)
           
                #get distributions
                mu  = self.fc_mu(x_encoded)
                logvar = self.fc_var(x_encoded)
                std=torch.sqrt(torch.exp(logvar))
                q_zx = torch.distributions.Normal(mu,std)
                p_z = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std)) #gaussienne
            
                #get latent vector
                z   = q_zx.rsample() #random vector from latent distributions
                #decode 
                x_logits = self.decoder(z)  # x hat is not an image but a parameter for the distribution. 
                x_hat = torch.sigmoid(x_logits)
                #reconstruction loss: 
                recon_loss = self.reconstruction_loss(x_logits,x)
                
                #kl divergence: 
                log_qzx = q_zx.log_prob(z)
                log_pz = p_z.log_prob(z)
                kl = log_qzx-log_pz
                kl = kl.sum(dim=-1).mean()

                #elbo loss: 
                elbo = kl+recon_loss
                epoch_test_loss += elbo.item()


        if epoch % 10 == 0:
            self.show_image(x,x_logits, z,epoch)
        
        return epoch_train_loss/len(train_dataloader),epoch_test_loss/len(test_dataloader),mu,std

    def __weights_histo(self,epoch):
        for name, param in self.named_parameters(): 
            writer.add_histogram(name, param, global_step=epoch, bins='tensorflow')




    def fit(self, traindata,testdata,validation_data=None, start_epoch=0, n_epochs=1000, lr=0.001, verbose=10,ckpt=None):
        # ic(lr)
        
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
            epoch_train_loss,epoch_test_loss,mu,std = self.__train_test_epoch(traindata,testdata,epoch)
            self.mu  = mu
            self.std = std
            print(f'\n Epoch {epoch+1} \n',
                        f'Train Loss= {epoch_train_loss:.4f}\n',f'test Loss= {epoch_test_loss:.4f}\n')
            
            
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/test',epoch_test_loss,epoch)
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


        
    def show_image(self,x,x_logits,z,epoch): 
        x_hat =  torch.sigmoid(x_logits)
        z_img = make_grid(z[:32].reshape(-1, 1, 1, self.dim_latent).repeat(1, 3, 1, 1))
        writer.add_image(f'{epoch}_latent_vectors', z_img, epoch)
        images_original = make_grid(x[:32].reshape(-1, 1, 28, 28).repeat(1, 3, 1, 1))
        images_predict = make_grid(x_hat[:32].reshape(-1, 1, 28, 28).repeat(1, 3, 1, 1))
        writer.add_images(f'{epoch}_images', torch.stack((images_original, images_predict)),
                                          epoch)

                                
       




VAE = VAE(dim_latent=5,dim_hidden=30,dim_out=30,device = device,opt=opt,weight_decay=0.004,vae= 'MLP')
VAE.fit(train_dataloader,test_dataloader,n_epochs=100,lr = 0.001)

VAE.show_image()


