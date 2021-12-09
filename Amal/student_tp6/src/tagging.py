import itertools
import logging
from numpy import hypot
from tqdm import tqdm
import datetime
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from icecream import ic
import torch.optim as optim
import os 
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)

ds = prepare_dataset('org.universaldependencies.french.gsd')

writer = SummaryWriter("runs/tag-"+time.asctime())

# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

train_data = TaggingDataset(ds.train, words, tags, True)
ic(len(train_data))
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)





#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)

# def train
#mettre tout ca dans une classe. 

class Training_Pytorch(nn.Module):
    def __init__(self,criterion,opt,embedder,decoder,rnn,ckpt_save_path=None):
        super().__init__()
        self.rnn = rnn
        self.optimizer = opt
        self.state={}
        self.embedder  = embedder
        self.decoder   = decoder
        self.ckpt_save_path = ckpt_save_path
        self.criterion=criterion

    def train_test_epoch(self,traindata,testdata,batch_size,hidden_size,nb_classes,nb_oov):
        epoch_train_loss = 0
        epoch_train_acc  = 0 
        epoch_test_loss  = 0
        epoch_test_acc   = 0 

        #train
        for idx,batch in enumerate(traindata):
            input,tags=batch
            self.opt.zero_grad()
            length, batch_size = input.shape
            if batch_size==BATCH_SIZE: #because for last batch, length size= 49 and i don't know how to fix this. 
                h0 = torch.zeros(1, batch_size, hidden_size).float() # not useful apparently maybe generated automatically ?  
                c0 = torch.zeros(1, batch_size, hidden_size).float() # not useful apparently maybe generated automatically ?  

                #take into account the OOV: on peut remplacer pendant l’apprentissage au hasard dans les 
                # exemples des mots par des mots OOV (en utilisant un token spécial -[OOV]).
                all_sentences=input.flatten()
                high=all_sentences.shape[0]
                oov_idx=torch.randint(low=0,high=high,size=(nb_oov,)) #create the index for the oov token.  #can use the nb of 
                all_sentences[oov_idx]=Vocabulary.OOVID

                #create the embedding: 
                input = all_sentences.view(length,batch_size)
                # ic(input.shape)
                input_embedded=self.embedder(input) #shape: len x batch_size x hidden_size 

                #use the model rnn to have the ouput and the states :
                output,(h_n,c_n) = self.rnn(input_embedded)

                #on calcule l'accuracy: 
                output = self.decoder(output)  #shape: length, batch_size, nb_classes
                prediction = torch.argmax(output,dim=2)
                prediction_no_0= prediction[torch.where(tags != 0)]
                tags_unpadded  = tags[torch.where(tags != 0 )]
                correct = (prediction_no_0 == tags_unpadded).sum().item()
                total = tags_unpadded.size(0)
                epoch_train_acc += correct/total

                #entrainement et calcul de la loss
                #il faut reshape output et tags car la cross entropy accepte des Inputs de size (N, C)
                # pour obtenir la bonne shape on doit passer output dans un decoder 
                output=output.view(-1,nb_classes)
                tags=tags.view(-1)
                loss = self.criterion(output,tags)
                loss.backward()
                self.opt.step()
                epoch_train_loss += loss.item()
            else: 
                pass


        #test 
        with torch.no_grad():
            for idx,batch in enumerate(testdata):
                input,tags=batch
                length, batch_size = input.shape
                if batch_size==BATCH_SIZE:
                    input_embedded=self.embedder(input) #shape: len x batch_size x hidden_size 
                    #Use the model rnn to have the ouput and the states :
                    output,(h_n,c_n) = self.rnn(input_embedded)
                    output = self.decoder(output)  #shape: length, batch_size, nb_classes 
                    #Compute accuracy during test : 
                    prediction = torch.argmax(output,dim=2)
                    prediction_no_0= prediction[torch.where(tags != 0)]
                    tags_unpadded  = tags[torch.where(tags != 0 )]
                    correct = (prediction_no_0 == tags_unpadded).sum().item()
                    total = tags_unpadded.size(0)
                    epoch_test_acc += correct/total
                    #loss: 
                    output=output.view(-1,nb_classes)
                    tags=tags.view(-1)
                    loss = self.criterion(output,tags)
                    epoch_test_loss += loss.item()
                else: 
                    pass
                            


        return epoch_train_loss/len(traindata), epoch_train_acc/len(traindata),epoch_test_loss/len(testdata),epoch_test_acc/len(testdata)
    
    


    def fit(self, traindata,testdata,batch_size,hidden_size,nb_classes,nb_oov,validation_data=None, start_epoch=0, n_epochs=25, lr=0.001, verbose=10,ckpt=None):
        parameters = list(self.embedder.parameters())+list(self.decoder.parameters())+list(self.rnn.parameters())
        if self.optimizer=="SGD":
            self.opt = optim.SGD(parameters,lr=lr,momentum=0.9)
        if self.optimizer=='Adam':
            self.opt = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
                g['lr'] = state['lr']
            
        for epoch in range(start_epoch,n_epochs):
            epoch_train_loss,epoch_train_acc,epoch_test_loss,epoch_test_acc = self.train_test_epoch(traindata,testdata,batch_size,hidden_size,nb_classes,nb_oov)
            print(f'\n Epoch {epoch+1} \n',
                        f'Train Loss= {epoch_train_loss:.4f}\n',f'Train Acc={epoch_train_acc:.4f}\n',f'test Loss= {epoch_test_loss:.4f}\n',f'Test Acc={epoch_test_acc:.4f}\n')

            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/test',epoch_test_loss,epoch)
            writer.add_scalar('Acc/train',epoch_train_acc,epoch)
            writer.add_scalar('Acc/test',epoch_test_acc,epoch)
            
            if self.ckpt_save_path:
                self.state['lr'] = lr
                self.state['epoch'] = epoch
                self.state['state_dict'] = self.state_dict()
                if not os.path.exists(self.ckpt_save_path):
                    os.mkdir(self.ckpt_save_path)
                torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{start_time}_epoch{epoch}.ckpt'))


    

    def predict(self,testdata):
        #choose a random phrase and token 
        idx = torch.randint(low=0,high=10,size=(1,)) #only choose one phrase for now
        phrase, token = testdata[idx]
        phrase = torch.tensor(phrase)
        phrase=  phrase.unsqueeze(1)
        token   = torch.tensor(token)
        
        input_embedded = self.embedder(phrase)
        output, (hn, cn) = self.rnn(input_embedded)  #output: (len x batch x hidden_size)
        tags_l = self.decoder(output)
        predictions = torch.argmax(tags_l, dim=2)


        #print the prediction: 
        #on print la phrase avec les tags:
        phrase = words.getwords(phrase)
        tokens = [j.item()  for j in token] #on transforme les tags en liste
        prediction = [pred.item() for pred in predictions]
        for i in range(len(phrase)):
            print(f'phrase= {phrase[i]}\n',f'prediction={prediction[i]}\n',f'tags= {tokens[i]}\n')
            #print with ground truth: 
            print(f'phrase= {phrase[i]}\n',f'prediction={tags.id2word[prediction[i]]}\n',f'tags= {tags.id2word[tokens[i]]}\n')
        

        
        

embedding_size=100
nb_classes=len(tags.word2id)
num_layers = 2
hidden_size=100
nb_oov = 100
lr=0.001
embedder = torch.nn.Embedding(num_embeddings = len(words),embedding_dim=embedding_size)
rnn = torch.nn.LSTM(input_size = embedding_size,hidden_size = hidden_size, num_layers=num_layers,batch_first = False) # a verifier si cest pas plutot false
criterion = nn.CrossEntropyLoss(ignore_index=0)
decoder =  nn.Linear(in_features=hidden_size, out_features=nb_classes)

training = Training_Pytorch(criterion=criterion,opt='Adam',embedder=embedder,decoder=decoder,rnn=rnn,ckpt_save_path='./models/tagging')
loss = training.fit(train_loader,test_loader,batch_size=BATCH_SIZE,hidden_size=hidden_size,nb_classes=nb_classes,nb_oov=nb_oov)
predict = training.predict(test_data)


