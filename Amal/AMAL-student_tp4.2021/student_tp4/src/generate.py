import string
from typing import List
import unicodedata
import torch
import sys
import torch.nn as nn
from icecream import ic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

import torch.nn.functional as F
from utils import  device
from model import RNN

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))

id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))
ic(len(lettre2id))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([(lettre2id[c]) for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
       
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]

        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
            # ic(self.phrases)
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self,i):
 
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        t = t.reshape(-1,1)
        return t[:-1],t[1:]




#  TODO: 
#Load data: 
txt = open('/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/AMAL-student_tp4.2021/student_tp4/src/nicolas_work/Nicolas_Olivain_tp4/data/trump_full_speech.txt', 'r').read()
batch_size=1000
nb_epochs=100
train_dataset = TrumpDataset(txt,maxsent=300)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=False)
criterion = torch.nn.CrossEntropyLoss()


train_features,train_label=next(iter(train_dataloader))


model     = RNN(in_features=1,hidden_size=30,out_features=len(id2lettre),mode='generation',criterion=criterion,opt='SGD',model='many to many',ckpt_save_path='./ckpt/Trump/hidden20') #output=10 car il y a 10 stations pour l'isntant
model.fit(train_dataloader, n_epochs=100, lr=0.001, verbose=1)
liste=model.predict(train_features[:,1,:], sequence_length=10)
print(liste)


          












