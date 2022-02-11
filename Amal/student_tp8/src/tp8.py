import logging
from re import S

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)
import os
from pathlib import Path
import heapq
from pathlib import Path
import gzip
from icecream import ic
import datetime
from tqdm import tqdm
import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset, random_split,Subset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
from tp8_preprocess import TextDataset

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire

vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer


tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=64
conv  = False
dummy = True
nb_classes = 3

# --- Chargements des jeux de données train, validation et test

TRAIN_RATIO = 0.005
new_len_train = int(TRAIN_RATIO * len(train))
# ic(new_len_train)
val_size = 1000
# train_size = len(train) - val_size
train_size = int(TRAIN_RATIO * len(train)) 

print('stop')

# train, val = random_split(train, [train_size, val_size])
train, val = random_split(dataset=train,lengths=[int(new_len_train), int(len(train) - new_len_train)], generator=torch.Generator().manual_seed(42069))



logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_dataloader = DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_dataloader = DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_dataloader = DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)

input,label=next(iter(train_dataloader))
ic(input.shape)
ic(label.shape)





#  TODO: 

#mettre un global pooling (max sur chaque channel puis local pooling) puis mettre un fully connected.

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
            
            # input = input.reshape(-1,self.input_size)
            b_size,n_features = input.shape
            if b_size==TRAIN_BATCHSIZE:
                output = self(input) # self c'est le modèle
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
            for idx, data in enumerate(testdata):
                input,label = data
                # ic(input.shape)
                # ic(label.shape)
                b_size,n_features = input.shape
                # ic(b_size)
                if b_size==TEST_BATCHSIZE:
                    output = self(input) # self c'est le modèle
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
        # ic(len(dataloader))
        for idx, data in enumerate(dataloader):
            self.opt.zero_grad()
            input,label = data
            
            # input = input.reshape(-1,self.input_size)
            b_size,n_features = input.shape
            if b_size==TEST_BATCHSIZE:
                output = self(input) # self c'est le modèle
                loss = self.criterion(output,label)
                n_correct = (torch.argmax(output,dim=1)==label).sum().item()
                total     = label.size(0)
                epoch_acc += n_correct/total
                epoch_loss += loss.item()

                writer.add_scalar('Loss/train_idx',loss,idx)
            
        return epoch_loss/len(dataloader), epoch_acc/len(dataloader)


    def fit(self, traindata,testdata,validation_data=None,batch_size=300, start_epoch=0, n_epochs=1000, lr=0.001, verbose=10,ckpt=None):
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

 

class DummyNet(training_global):
    def __init__(self,input_size,hidden_dim,emb_dim,output_size,criterion,device,opt,dropout, regularization=None,ckpt_save_path=None,weight_decay=0):
        super(DummyNet,self).__init__(regularization = regularization,criterion =criterion,device = device,opt =  opt,ckpt_save_path= ckpt_save_path,weight_decay=weight_decay)
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_dim=hidden_dim
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.decoder =  nn.Linear(in_features=emb_dim, out_features=nb_classes) #3 classes
        self.globalpooling = nn.AdaptiveMaxPool2d((emb_dim,1))

    def forward(self,x):
        x = self.embedding(x) 
        out = self.globalpooling(x)
        # ic(out.shape)
        out = out.squeeze(2)
        out = self.decoder(out)

        # ic(out.shape)
        return out #no softmax because already included in the loss. 


class ConvNet(training_global):
    def __init__(self,input_size,hidden_dim,emb_dim,output_size,criterion,device,opt,dropout, regularization=None,ckpt_save_path=None,weight_decay=0):
        super(ConvNet,self).__init__(regularization = regularization,criterion =criterion,device = device,opt =  opt,ckpt_save_path= ckpt_save_path,weight_decay=weight_decay)
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_dim=hidden_dim
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim,hidden_dim,kernel_size  = 3, stride = 2,padding = 0 )
        self.conv2 = nn.Conv1d(hidden_dim,hidden_dim,kernel_size  = 3, stride = 2,padding = 0 )
        self.conv3 = nn.Conv1d(hidden_dim,output_size,kernel_size  = 3, stride = 2,padding = 0 )
        self.activation = nn.ReLU()
        self.globalpooling = nn.AdaptiveMaxPool2d((output_size,1))
        self.fc = nn.Linear(output_size,nb_classes) #3 sorties car on a neutral, negative et positif. 
        self.m = nn.MaxPool1d(3, stride=2)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,x):
        x = self.embedding(x)
        x = x.view(x.shape[0],x.shape[2],x.shape[1]) # to make x of shape: Batch, Channel, Lenght. 
        x = self.conv1(x)
        x = self.m(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        out = self.globalpooling(x) #global max pooling #input msut be N,C,L and output will be N,C,Lout
        # out = torch.tile(torch.max(x, dim=2)[0], (1, 1, 1, 5)) # global max pooling
        out = out.squeeze(2) # to add 1 at the last channel so it works with fully connected. 
        out = self.fc(out)
        return out



criterion = torch.nn.CrossEntropyLoss()


if dummy: 
    net = DummyNet(input_size=1,hidden_dim=100,output_size=100,emb_dim=100,criterion=criterion,device=device,regularization = None,dropout=0.3, opt='Adam',weight_decay = 0.009348818325128058)
    
if conv : 
    net = ConvNet(input_size=1,hidden_dim=100,output_size=100,emb_dim=100,criterion=criterion,device=device,regularization = None,dropout=0.3, opt='Adam',weight_decay = 0.009348818325128058)

net.fit(train_dataloader,test_dataloader,lr=0.001)
        