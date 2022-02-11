import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import datetime
import os
from icecream import ic
import math 
from torch import optim
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PAD_ID = 40001


class MLP(nn.Module):
    def __init__(self, dim_in, hidden_sizes, dim_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in,hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes,dim_out))
        
    def forward(self,x):
        return self.model(x)





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
    


def collate(batch):
    data = [torch.LongTensor(b[0]) for b in batch]
    lens = [len(b[0]) for b in batch]
    labels = [b[1] for b in batch]
    return torch.nn.utils.rnn.pad_sequence(data, padding_value = PAD_ID, batch_first=True), torch.LongTensor(lens), torch.LongTensor(labels)

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
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



dico, embeddings, train_data,test_data = get_imdb_data()


#  TODO: 

# # --- Chargements des jeux de données train, validation et test

nb_class = len(dico)
embedding_size = 50 
TRAIN_BATCHSIZE=64
TEST_BATCHSIZE=64
val_size = 1000
train_size = len(train_data) - val_size
train, val = random_split(train_data, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test_data))
# logging.info("Vocabulary size: %d", vocab_size)
train_dataloader = DataLoader(train, batch_size=TRAIN_BATCHSIZE,collate_fn = collate)
val_dataloader = DataLoader(val, batch_size=TEST_BATCHSIZE,collate_fn = collate)
test_dataloader = DataLoader(test_data, batch_size=TEST_BATCHSIZE,collate_fn = collate)

input,lens,label=next(iter(train_dataloader))



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
            input,lens,label = data
            
            # input = input.reshape(-1,self.input_size)
            b_size,n_features = input.shape
            if b_size==TRAIN_BATCHSIZE:
                output = self(input) # self c'est le modèle
                loss = self.criterion(output,label)
                n_correct = (torch.argmax(output,dim=1)==label).sum().item()
                total     = label.size(0)
                epoch_train_acc += n_correct/total
                loss.backward()
                self.opt.step()
                epoch_train_loss += loss.item()

        with torch.no_grad():
            for idx, data in enumerate(testdata):
                input,lens,label = data
     
                b_size,n_features = input.shape
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


#masque sur l'attention de la collate fn: 

class SimpleNet(training_global):
    def __init__(self,weights_embeddings,dim_h,criterion,device,opt,ckpt_save_path=None,weight_decay=0):
        super(SimpleNet,self).__init__(criterion = criterion,device = device,opt = opt,ckpt_save_path= ckpt_save_path,weight_decay=weight_decay)
        weights_embeddings = torch.Tensor(weights_embeddings)
        _, dim_emb = weights_embeddings.shape
        self.embedded   = nn.Embedding.from_pretrained(weights_embeddings) #mettre une liste
        self.classifier = nn.Sequential(nn.Linear(dim_emb,dim_h),nn.ReLU(),nn.Linear(dim_h,2))


    def forward(self,x):
        emb = self.embedded(x)
        out = emb.mean(dim=1)
        logits = self.classifier(out)

        return logits


class QueryNet(training_global):
    def __init__(self,weights_embeddings,dim_h,criterion,device,opt,ckpt_save_path=None,weight_decay=0):
        super(QueryNet,self).__init__(criterion = criterion,device = device,opt = opt,ckpt_save_path= ckpt_save_path,weight_decay=weight_decay)
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
        attent_logits = torch.matmul(q,k.transpose(-2,-1))  #
        attent_logits = attent_logits/math.sqrt(d)
        if mask is not None : 
            attent_logits = attent_logits.masked_fill(mask ==0 ,-1e9)
        attention = F.softmax(attent_logits,dim=-1)

        attention = self.dropout(F.softmax(attention, dim=-1)) 
        output = torch.matmul(attention, v) 
        return output,attention



class OneHeadAttention(nn.Module): 
    def __init__(self,embedding_dim,dim_out):
        super(OneHeadAttention,self).__init__()
        self.query = nn.Linear(embedding_dim,dim_out)
        self.value = nn.Linear(embedding_dim,dim_out)
        self.key   = nn.Linear(embedding_dim,dim_out)
        self.linear = nn.Sequential(
            nn.Linear(dim_out,dim_out),
            nn.ReLU()
            )
        self.scaleproduct = ScaledDotProductAttention()

    def forward(self,x):
        q = self.query(x)
        v = self.value(x)
        k = self.key(x)
        output, attention = self.scaleproduct(q,v,k)
        output = self.linear(output)
        # output = F.relu(output)
        return output




class SelfAttentionModel(training_global):
    def __init__(self,weights_embeddings,criterion,device,opt,ckpt_save_path=None,weight_decay=0):
        super(SelfAttentionModel,self).__init__(criterion = criterion,device = device,opt = opt,ckpt_save_path= ckpt_save_path,weight_decay=weight_decay)
        weights_embeddings = torch.Tensor(weights_embeddings)
        _, dim_emb = weights_embeddings.shape
        self.embedded        = nn.Embedding.from_pretrained(weights_embeddings)
        self.selfattention1  = OneHeadAttention(dim_emb,dim_emb)
        self.ln1             = nn.LayerNorm(dim_emb)
        self.selfattention2  = OneHeadAttention(dim_emb,dim_emb)
        self.ln2             = nn.LayerNorm(dim_emb)
        self.selfattention3  = OneHeadAttention(dim_emb,dim_emb)
        self.ln3             = nn.LayerNorm(dim_emb)

        self.classifier      = nn.Linear(dim_emb,2)
        

    def forward(self,x): 
        emb = self.embedded(x)
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



#training: 

criterion = torch.nn.CrossEntropyLoss()

net = SelfAttentionModel(weights_embeddings=embeddings,criterion=criterion,device=device,opt='Adam')
# net = SimpleNet(weights_embeddings=embeddings,dim_h=100,criterion=criterion,device=device,opt='Adam')
# net = QueryNet(weights_embeddings=embeddings,dim_h=100,criterion=criterion,device=device,opt='Adam')
net.fit(train_dataloader,test_dataloader,lr= 0.005,n_epochs=10)




# comment faire dépendre q du texte : on met pas d,1 mais d,d. 


# pour le faire avec un RNN on porte l'attention sur les états cachés. au lieu de porter l'attentins ur les token, les token deviennent les etat caché au bout de un état, deux états trois états. 


