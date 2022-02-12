
from cgi import test
import math
import click
from torch.utils.tensorboard import SummaryWriter
import logging

from typing import Optional
import re
from icecream import ic
import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import os
from torch import optim
from datamaestro import prepare_dataset
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import torch.nn as nn
from utils import PositionalEncoding
from torch.utils.data import Dataset, DataLoader
from Lightning_utils import LightningNetwork


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_ID = 40001
MAX_LENGTH = 500

embedding_size = 50 


logging.basicConfig(level=logging.INFO)

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]
    def get_txt(self,ix):
        s = self.files[ix]
        return s if isinstance(s,str) else s.read_text(), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300])

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText) train
    - DataSet (FolderText) test

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        'edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")
    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

#  TODO: 

class MLP(nn.Module):
    def __init__(self, dim_in, hidden_sizes, dim_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in,hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes,dim_out))
        
    def forward(self,x):
        return self.model(x)




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self,  attn_dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self,q,k,v,mask = None):
        """
        parameters : 
        mask : represents the optional masking of specific entries in the attention matricx
        returns : 
        values : output of the attention layers
        attention : attention weight"""
        d = q.size()[-1] #emb_di
        attent_logits = torch.bmm(q,k.transpose(-2,-1))  #
        attent_logits = attent_logits/math.sqrt(d)
        if mask is not None : 
            attent_logits = attent_logits.masked_fill(mask ==0 ,-1e9)
        attention = F.softmax(attent_logits,dim=-1)

        attention = self.dropout(F.softmax(attention, dim=-1)) 
        output = torch.bmm(attention, v) 
        return output,attention

class SimpleNet(nn.Module):
    def __init__(self,weights_embeddings,dim_h):
        super(SimpleNet,self).__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)
        _, dim_emb = weights_embeddings.shape
        self.embedded   = nn.Embedding.from_pretrained(weights_embeddings) #mettre une liste
        self.classifier = nn.Sequential(nn.Linear(dim_emb,dim_h),nn.ReLU(),nn.Linear(dim_h,2))


    def forward(self,x):
        emb = self.embedded(x)
        out = emb.mean(dim=1)
        logits = self.classifier(out)

        return logits


class QueryNet(nn.Module):
    def __init__(self,weights_embeddings,dim_h):
        super(QueryNet,self).__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)
        _, dim_emb = weights_embeddings.shape
        self.embedded   = nn.Embedding.from_pretrained(weights_embeddings) #mettre une liste
        self.query      = nn.Linear(dim_emb,1) #si query = 1 alors ils vont tous avoir la même query, par contre si on met en sortie d au lieu de 1 alors on a un alpha pour chaque mot. 
        self.classifier = nn.Sequential(nn.Linear(dim_emb,dim_h),nn.ReLU(),nn.Linear(dim_h,2))


    def forward(self,x):
        emb = self.embedded(x)
        alpha = torch.where((x == PAD_ID).unsqueeze(dim=2), torch.tensor(-np.inf, dtype=torch.float),
                            self.query(emb)) # pour créer le alpha sauf sur le padding token. 
        alpha = F.softmax(alpha,dim=1) 
        #on modifie la shape de alpha pour copier chaque truc sur la dim embedding: 
        alpha = alpha.expand(emb.shape)
        # ic(alpha.shape)
        out = (alpha * emb).sum(dim=1) 
        # ic(out.shape)
        logits = self.classifier(out)

        return logits





class OneHeadAttention(nn.Module):
    def __init__(self,
                 qk_dim,
                 v_dim,
                 embedding_dim,
                 out_dim
                 ) -> None:
        super().__init__()
        self.linear_q = nn.Linear(embedding_dim, qk_dim)
        self.linear_v = nn.Linear(embedding_dim, v_dim)
        self.linear_k = nn.Linear(embedding_dim, qk_dim)
        self.linear_o = nn.Linear(v_dim, out_dim)
        self.dotproduct = ScaledDotProductAttention()

    def forward(self, x):
        q = self.linear_q(x)    # (batch_size, length, qk_dim)
        v = self.linear_v(x)    # (batch_size, length, v_dim)
        k = self.linear_k(x)      # (batch_size, length, qk_dim)
        out, attention = self.dotproduct(q, k, v)
        out = self.linear_o(out)
        out = F.relu(out)

        return out


class BasicSelfAttentionModel(nn.Module):
    """simple self attention model """
    def __init__(self,qk_dim,v_dim,out_dim,dim_emb,class_dim,weights_embeddings):
        super(BasicSelfAttentionModel,self).__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)
        # _, dim_emb = weights_embeddings.shape
        dim_emb = dim_emb
        self.embedded        = nn.Embedding.from_pretrained(weights_embeddings)
        self.selfattention1  = OneHeadAttention(qk_dim,v_dim,dim_emb,out_dim)
        self.ln1             = nn.LayerNorm(out_dim)
        self.selfattention2  = OneHeadAttention(qk_dim,v_dim,dim_emb,out_dim)
        self.ln2             = nn.LayerNorm(out_dim)
        self.selfattention3  = OneHeadAttention(qk_dim,v_dim,dim_emb,out_dim)
        self.ln3             = nn.LayerNorm(out_dim)
        self.classifier      = nn.Linear(out_dim,class_dim)
        
    def forward(self,x):
        emb = self.embedded(x)
        att1 = self.selfattention1(emb)
        att1 = self.ln1(att1)
        att2 = self.selfattention2(att1)
        att2 = self.ln2(att2)
        att3 = self.selfattention3(att2)
        out = self.ln3(att3)
        out = out.mean(dim=1)
        out = self.classifier(out)
        return out 


class SelfAttentionModel(nn.Module):
    "residual + (optional)CLS token + positional encoding self attention model"
    def __init__(self,qk_dim,v_dim,out_dim,class_dim,dim_emb,max_len,weights_embeddings,pad_idx,  use_cls = None) -> None:
        super(SelfAttentionModel,self).__init__()
        weights_embeddings = torch.Tensor(weights_embeddings)
        # _, dim_emb = weights_embeddings.shape
        dim_emb = dim_emb
        self.pad_idx = pad_idx
        self.cls_idx = pad_idx + 1 
        self.embedded        = nn.Embedding.from_pretrained(weights_embeddings,padding_idx = self.pad_idx)
        self.selfattention1  = OneHeadAttention(qk_dim,v_dim,dim_emb,out_dim)
        self.ln1             = nn.LayerNorm(out_dim)
        self.selfattention2  = OneHeadAttention(qk_dim,v_dim,dim_emb,out_dim)
        self.ln2             = nn.LayerNorm(out_dim)
        self.selfattention3  = OneHeadAttention(qk_dim,v_dim,dim_emb,out_dim)
        self.ln3             = nn.LayerNorm(out_dim)
        self.pos_enc = PositionalEncoding(dim_emb,max_len=max_len)  
        self.classifier      = nn.Linear(out_dim,class_dim)
        self.use_cls = use_cls if not None else None 
        if self.use_cls is not None : 

            self.linear_cls = nn.Linear(dim_emb, dim_emb)
        

    def forward(self,x): 
        if self.use_cls is not None:

            batch_size, _ = x.size()
            cls_token = (torch.ones(batch_size,1) * self.cls_idx).long().to(x.device)
            input = torch.cat([cls_token, x], dim=1)
        emb = self.embedded(x)
        if self.use_cls:
     
            emb[:, 0,:] = self.linear_cls(torch.ones_like(emb[:, 0,:]).to(emb.device))
        emb = self.pos_enc(emb)
        att1 = self.selfattention1(emb)
  
        out1 = self.ln1(emb + att1)
        att2 = self.selfattention2(out1)
        out2 = out1 + att2
        out2 = self.ln2(out2)
        att3 = self.selfattention3(out2)
        out3 =  out2 + att3
        out = self.ln3(out3)
        out = out.mean(dim=1) #on fait la moyenne sur le nombre de mots , 
        out = self.classifier(out)

        return out 



        

@click.command()
@click.option('--test-iterations', default=1000, type=int, help='Number of training iterations (batches) before testing')
@click.option('--epochs', default=50, help='Number of epochs.')
@click.option('--modeltype', required=True, type=int, help="0: base, 1 : Attention1, 2: Attention2")
@click.option('--emb-size', default=100, help='embeddings size')
@click.option('--batch-size', default=20, help='batch size')
def main(epochs,test_iterations,modeltype,emb_size,batch_size):
    conf = OmegaConf.load('./config/conf.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
    nb_class = len(word2id)
    id2word = dict((v, k) for k, v in word2id.items())
    PAD = word2id["__OOV__"]
    embeddings = torch.Tensor(embeddings)
    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embeddings))

    # def collate(batch):
    #     """ Collate function for DataLoader """
    #     data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
    #     lens = [len(d) for d in data]
    #     labels = [item[1] for item in batch]
    #     return emb_layer(torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD)).to(device), torch.LongTensor(labels).to(device), torch.Tensor(lens).to(device)
    
    def collate(batch):
        """ Collate function for DataLoader """
        data = [torch.LongTensor(item[0][:MAX_LENGTH]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        return torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD), torch.LongTensor(labels)

    train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=batch_size,collate_fn=collate,shuffle=False)
    
    ##  TODO: 
    if modeltype == 0:
        network = SimpleNet(weights_embeddings=embeddings,dim_h=100)
        # network = QueryNet(weights_embeddings=embeddings,dim_h=100)
        name = "BasicEncoder"
    if modeltype == 1:
        network = BasicSelfAttentionModel(qk_dim=conf.qk_dim,v_dim=conf.v_dim,out_dim=conf.out_dim,dim_emb=conf.dim_emb,class_dim=conf.class_dim,weights_embeddings=embeddings)
        name = "BasicEncoder"
    if modeltype == 2:
        # network = AdvancedResEncoder(qk_dim=conf.qk_dim, v_dim=conf.v_dim, embedding_w=embeddings, max_len=conf.max_len, use_cls=conf.use_cls,
            #  emb_freeze=conf.emb_freeze, pad_idx=PAD, out_dim=conf.out_dim, class_dim=conf.class_dim, dropout_prob=conf.dropout_prob)
        network = SelfAttentionModel(qk_dim=conf.qk_dim,v_dim=conf.v_dim,out_dim=conf.out_dim,dim_emb=conf.dim_emb,class_dim=conf.class_dim,max_len= conf.max_len,weights_embeddings=embeddings,pad_idx =PAD,use_cls = True)
        name = "AdvancedResEncoder"
    model = LightningNetwork(name=name, network=network, learning_rate=0.001)

    LOG_PATH = Path('./logs') 
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode='max')
    logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,
                        logger=logger,
                        default_root_dir=LOG_PATH,
                        max_epochs=10,
                        callbacks=[checkpoint_callback])
    
    hyperparameters = conf
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_loader, test_loader)

   



if __name__ == "__main__":
    conf = OmegaConf.load('./config/conf.yaml')
    # main(epochs = 10,test_iterations=1000,modeltype=2,emb_size=50,batch_size=20)
    main()


