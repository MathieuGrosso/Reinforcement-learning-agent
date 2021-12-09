import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
import datetime
from pathlib import Path
from typing import List
import random 
# import sentencepiece as spm
import os 
from icecream import ic
import time
import re
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sentencepiece as spm

print(torch.cuda.is_available())

logging.basicConfig(level=logging.INFO)


FILE = "/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/student_tp6/data/en-fra.txt" # for local training
# FILE= "en-fra.txt" #for collab traning

writer = SummaryWriter("runs/trad-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))
vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)
train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)


# pré traitement et segmentation: 

spm.SentencePieceTrainer.train(
        input=FILE,
        pad_id=3,
        model_prefix='/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/student_tp6/src/vocab',
        vocab_size=5000
    )
sp = spm.SentencePieceProcessor(model_file='/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/student_tp6/src/vocab.model')



class TradDatasetNgram():
    def __init__(self,data,sp,adding=True,max_len=10):
        self.sentences = []
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor(sp.encode(orig, out_type=int, add_eos=True)), torch.tensor(sp.encode(dest, out_type=int, add_eos=True))))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self,i):
        return self.sentences[i]


def collate_ngram(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig,padding_value=sp.pad_id()),o_len,pad_sequence(dest,padding_value=sp.pad_id()),d_len


datatrain_ngram = TradDatasetNgram("".join(lines[:idxTrain]),sp,max_len=MAX_LEN)
datatest_ngram = TradDatasetNgram("".join(lines[idxTrain:]),sp,max_len=MAX_LEN)
train_loader_ngram = DataLoader(datatrain_ngram, collate_fn=collate_ngram, batch_size=BATCH_SIZE, shuffle=True)
test_loader_ngram = DataLoader(datatest_ngram, collate_fn=collate_ngram, batch_size=BATCH_SIZE, shuffle=True)




class train_global(nn.Module):
    def __init__(self,criterion,gru_encode,gru_decode,decoder,embedder_fr,embedder_ang,opt,device,teacher_forcing_ratio = 0.5,ckpt_save_path=None ):
        """A Sequence to Sequence network, or seq2seq network, or Encoder Decoder network, is a model consisting of 
        two RNNs called the encoder and decoder. The encoder reads an input sequence and outputs a single vector, and the decoder 
        reads that vector to produce an output sequence."""

        super().__init__()
        self.device = device
        self.state={}
        self.criterion = criterion
        self.gru_encode = gru_encode
        self.gru_encode.to(device)
        self.gru_decode = gru_decode
        self.gru_decode.to(device)
        self.decoder    = decoder
        self.decoder.to(device)
        self.embedder_fr = embedder_fr
        self.embedder_fr.to(device)
        self.embedder_ang = embedder_ang
        self.embedder_ang.to(device)
        self.opt = opt
        self.ckpt_save_path = ckpt_save_path
        self.teacher_forcing_ratio = teacher_forcing_ratio
        

    def EncoderRNN(self,input):
  
        input_embedded  = self.embedder_ang(input)

        output,hidden   = self.gru_encode(input_embedded) #output shape: 
        return output, hidden

    

    def train_test_epoch(self,traindata,testdata,encoder_opt,decoder_opt,batch_size,hidden_size,nb_classes,nb_oov):
        """does train and test of the model"""
        """teacher forcing: is the concept of using the real target outputs as each next input,
         instead of using the decoder’s guess as the next input. Using teacher forcing causes it to converge faster but 
         when the trained network is exploited, it may exhibit instability."""
        
        epoch_train_loss = 0
        epoch_train_acc  = 0 
        epoch_test_loss  = 0
        epoch_test_acc   = 0 


        #train
        for idx,batch in enumerate(traindata):
            (phraseEng, phraseEng_len, phraseFra, phraseFra_len)=batch
            phraseEng=phraseEng.to(device)
            phraseFra=phraseFra.to(device)
            length,batch_size=phraseEng.shape
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()

            if batch_size==BATCH_SIZE:


                #take into account the OOv: 
                all_sentences_eng=phraseEng.flatten()
                high=all_sentences_eng.shape[0]
                oov_idx=torch.randint(low=0,high=high,size=(nb_oov,)) #create the index for the oov token.  #can use the nb of 
                all_sentences_eng[oov_idx]=Vocabulary.OOVID

                #do the encoding: 
                all_sentences_eng=all_sentences_eng.view(length,batch_size)
                output,hidden = self.EncoderRNN(all_sentences_eng)
               
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False


                #decoding: 
                """In the simplest seq2seq decoder we use only last output of the encoder called the context. 
                    This context vector is used as the initial hidden state of the decoder.
                    The initial input token is the start-of-string <SOS> token, and the first hidden state is the context vector ."""

                if use_teacher_forcing:

                    """train with teacher forcing:
                    teacher forcing: we use the next word of the sentence starting with SOS as input instead of using the prediction of the model. """
                    # do the decoding: 
                    decoder_h0=hidden # hidden start correspond to last hiddn of the encoder 
       
                    decoder_input = torch.zeros((1,batch_size),device=device)+vocFra.SOS
                    decoder_input=decoder_input.to(device)
                    decoder_input = torch.cat((decoder_input, phraseFra)) #prepare the input--> cest ce vecteur que l'on va passer. 
                    decoder_input=decoder_input.to(device)
                    input_embedded = self.embedder_fr(decoder_input.long())
                    output, hidden_next = self.gru_decode(input_embedded,decoder_h0) # the model predict output et on lui donne à chaque tour le reste de la phrase et pas l'output
                    output = self.decoder(output) #on utilise l'output à decoder car on est en teacher forcing. 
                    
                    
                    #compute target : add sos to phrasefra: 
                    sos = torch.zeros((1,batch_size),device=device)+vocFra.SOS
                    sos = sos.to(device)
                    phraseFra = torch.cat((sos,phraseFra))

                    #accuracy
                    prediction = torch.argmax(output,dim=2)
                    prediction_no_0= prediction[torch.where(phraseFra != vocFra.PAD)]
                    target_unpadded = phraseFra[torch.where(phraseFra!= vocFra.PAD )]
                    correct = (prediction_no_0 == target_unpadded).sum().item()
                    total = target_unpadded.size(0)
                    epoch_train_acc += correct/total

                    #loss:
                    output=output.view(-1,embedding_dim)
                    target = phraseFra.reshape(-1)
                    loss = self.criterion(output,target.long())
                    


                else: 
                    output_list = []
                    prediction_list = []
                    """le mode non contraint où la phrase cible n’est pas considérée lors de la génération itérative de la traduction : 
                    c’est le mot correspondant à la probabilité maximale du décodage de l’état latent du pas précédent qui est introduit à chaque 
                    pas de temps (ou un tirage aléatoire dans cette distribution); ce mode consiste à générer comme si vous étiez en inférence puis à corriger une fois toute la phrase engendrée."""
                    
                    ##pour le premier mot on utilise le token SOS puis après pour lessuivant on utilise l'output du modèle. 
                    decoder_h0=hidden # hidden start correspond to last hiddn of the encoder 
                    decoder_input = torch.zeros(size=(1,batch_size),device=device)+vocFra.SOS # on a besoin du token SOS pour commencer. 
                    decoder_input=decoder_input.to(device)
                    input_embedded = self.embedder_fr(decoder_input.long())
                    output, hidden_next = self.gru_decode(input_embedded,decoder_h0) # the model predict output et on lui donne à chaque tour le reste de la phrase et pas l'output
                    output = self.decoder(output)  
                    wordnext = torch.argmax(output,dim=2)
                    output_list.append(output)
                    prediction_list.append(wordnext)

                
                    ##mot suivant: 
                    lenseq = phraseFra.shape[0] - 1
                    for step in range(lenseq):
                  
                        wordnext_embedded = self.embedder_fr(wordnext) #shape: 1xbatch_sizex embedding size. 
                        output,hidden_next=self.gru_decode(wordnext_embedded,hidden_next)
                        output = self.decoder(output)
                        wordnext = torch.argmax(output,dim=2)  
                        output_list.append(output)
                        prediction_list.append(wordnext)
                    
                    output_tensor = torch.cat(output_list)
                    prediction_tensor = torch.cat(prediction_list)

                    #accuracy :
                    
                    prediction_no_0 = prediction_tensor[torch.where(phraseFra != vocFra.PAD)]
                    target_unpadded = phraseFra[torch.where(phraseFra != vocFra.PAD )]
                    correct = len(torch.where(prediction_no_0 == target_unpadded)[0])
                    total   = len(torch.where(phraseFra != vocFra.PAD)[0])

                    epoch_train_acc += correct/total

                    #loss :
                    output=output_tensor.view(-1,embedding_dim)
                    target = phraseFra.reshape(-1)
                    loss = self.criterion(output,target.long())
            
                loss.backward()

                encoder_opt.step()
                decoder_opt.step()
                epoch_train_loss += loss.item()

            else: 
                pass

        #test
        with torch.no_grad():
            for idx, batch in enumerate(testdata):
                
                
                (phraseEng, phraseEng_len, phraseFra, phraseFra_len)=batch
                phraseEng=phraseEng.to(device)
                phraseFra=phraseFra.to(device)
                length,batch_size=phraseEng.shape
               
            
                if batch_size==BATCH_SIZE:
                    all_sentences_eng = phraseEng
                    #do the encoding: 
                    output,hidden = self.EncoderRNN(all_sentences_eng)

                    output_test_list = []
                    prediction_test_list = []

                    decoder_h0=hidden
                    decoder_input = torch.zeros(size =(1,batch_size))*vocFra.SOS
                    decoder_input=decoder_input.to(device)
                    decoder_h0.to(device)
                    input_embedded = self.embedder_fr(decoder_input.long())
                    output,hidden_next = self.gru_decode(input_embedded,decoder_h0)

                    output = self.decoder(output)
                    wordnext = torch.argmax(output,dim=2)
                    output_test_list.append(output)
                    prediction_test_list.append(wordnext)

                    lenseq = phraseFra.shape[0] - 1
                    for step in range(lenseq):
                        wordnext_embedded = self.embedder_fr(wordnext) #shape: 1xbatch_sizex embedding size. 
                        output,hidden_next=self.gru_decode(wordnext_embedded,hidden_next)
                        output = self.decoder(output)
                        wordnext = torch.argmax(output,dim=2)  
                        output_test_list.append(output)
                        prediction_test_list.append(wordnext)

                    output_tensor = torch.cat(output_test_list)
                    prediction_tensor = torch.cat(prediction_test_list)

                    #accuracy :
                    prediction_no_0 = prediction_tensor[torch.where(phraseFra != vocFra.PAD)]
                    target_unpadded = phraseFra[torch.where(phraseFra != vocFra.PAD )]
                    nb_truepos_test = len(torch.where(prediction_no_0 == target_unpadded)[0])
                    nb_samples_test = len(torch.where(phraseFra != vocFra.PAD)[0])     
                    test_acc =        nb_truepos_test/nb_samples_test    

                    epoch_test_acc += nb_truepos_test/nb_samples_test


                    #loss test: 
                    output=output_tensor.view(-1,embedding_dim)
                    target = phraseFra.reshape(-1)
                    
                    loss = self.criterion(output,target.long())
                    epoch_test_loss+=loss
                else: 
                    pass

                    


                



        return epoch_train_loss/len(traindata),epoch_train_acc/len(traindata),epoch_test_loss/len(testdata),epoch_test_acc/len(testdata)
                
    def fit(self,traindata,testdata,encode_params,decode_params,batch_size,hidden_size,nb_classes,nb_oov,test_data=None,validation_data=None, start_epoch=0, n_epochs=50, lr=0.001, verbose=10,ckpt=None):
        encode_params = list(self.embedder_ang.parameters()) + list(self.gru_encode.parameters())
        decode_params = list(self.embedder_fr.parameters())  + list(self.gru_decode.parameters()) + list(self.decoder.parameters())

        
        if self.opt=="SGD":
            encoder_optimizer = optim.SGD(encode_params, lr=lr) 
            decoder_optimizer = optim.SGD(decode_params, lr=lr)
        if self.opt=='Adam':
            encoder_optimizer = torch.optim.Adam(encode_params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            decoder_optimizer = torch.optim.Adam(decode_params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
                g['lr'] = state['lr']
            
        for epoch in range(start_epoch,n_epochs):
            epoch_train_loss,epoch_train_acc,epoch_test_loss,epoch_test_acc = self.train_test_epoch(traindata,testdata,encoder_opt=encoder_optimizer,decoder_opt=decoder_optimizer,batch_size=batch_size,hidden_size=hidden_size,nb_classes=nb_classes,nb_oov=nb_oov)
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/test',epoch_test_loss,epoch)
            writer.add_scalar('Acc/train',epoch_train_acc,epoch)
            writer.add_scalar('Acc/test',epoch_test_acc,epoch)

            print(f'\n Epoch {epoch+1} \n',
                        f'\n Train Loss= {epoch_train_loss:.4f}\n',f'\n Train Acc= {epoch_train_acc:.4f}\n',f'\n Test Loss= {epoch_test_loss:.4f}\n',f'\n Test Acc= {epoch_test_acc:.4f}\n')
        
            if self.ckpt_save_path:
                self.state['lr'] = lr
                self.state['epoch'] = epoch
                self.state['state_dict'] = self.state_dict()
                if not os.path.exists(self.ckpt_save_path):
                    os.mkdir(self.ckpt_save_path)
                torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{start_time}_epoch{epoch}.ckpt'))

                 
    def generate_greedy(self,testdata,lenseq=20):
        """we start by generating in a greedy way: we take the argmax at each step.  """
        with torch.no_grad():
            output_list = []
            phraseFra_pred = []
            idx = torch.randint(low=0,high=10,size=(1,)) #only choose one phrase for now
            phraseEng, phraseFra = testdata[idx]
            
        
            phraseEng = torch.tensor(phraseEng)
            phraseFra = torch.tensor(phraseFra)
            phraseEng = phraseEng.to(device)
            phraseFra = phraseFra.to(device)

            input = phraseEng
            input = input.unsqueeze(1)
            length,batch_size = input.shape
            input = input.to(device)
            output,hidden = self.EncoderRNN(input)


            #premier decodage
            decoder_h0=hidden # hidden start correspond to last hiddn of the encoder 
            decoder_input = torch.zeros(size=(1,1))*vocFra.SOS # on a besoin du token SOS pour commencer. 
            decoder_input = decoder_input.to(device)
            decoder_h0 = decoder_h0.to(device)
            input_embedded = self.embedder_fr(decoder_input.long())
            input_embedded = input_embedded.to(device)
            output, hidden_next = self.gru_decode(input_embedded,decoder_h0)
             # the model predict output et on lui donne à chaque tour le reste de la phrase et pas l'output
            output = self.decoder(output)  #shape length x batch_size x num classes --> to verify. 
            wordnext = torch.argmax(output,dim=2)       
            output_list.append(output)
            phraseFra_pred.append(wordnext)
            for step in range(max(length-1,lenseq)):
                wordnext_embedded = self.embedder_fr(wordnext) #shape: 1xbatch_sizex embedding size. 
                output,hidden_next=self.gru_decode(wordnext_embedded,hidden_next)
                output = self.decoder(output)
                wordnext = torch.argmax(output,dim=2)  #return the indices of the max value of all elements in the input tensor
                output_list.append(output)
                phraseFra_pred.append(wordnext)
                if wordnext == vocFra.EOS:
                    break 
            phraseFra_hat = torch.cat(phraseFra_pred).reshape(-1)
            print('Phrase anglais:', vocEng.getwords(phraseEng))
            print('Phrase français (truth):', vocFra.getwords(phraseFra))
            print('Traduction:', vocFra.getwords(phraseFra_hat))

    


    def generate_last_beam(self,testdata,k=3):
        str = vocFra.SOS  #initialisation de la str avec le caractère start
        k_strings = [[] for i in range(k)]  #initialise les k string à la valeur donnée en entrée
        k_probas = [0] * k #initialise les k logprobas à 0
        k_output=[]
        k_hidden=[]
        nb_char = 0
        sequences = [[list(),0.0]]
        prob=[]
        indx=[]

        with torch.no_grad():
            
            idx = torch.randint(low=0,high=10,size=(1,)) #only choose one phrase for now
            phraseEng, phraseFra = testdata[idx]
            phraseEng = torch.tensor(phraseEng)
            phraseFra = torch.tensor(phraseFra)
            phraseEng = phraseEng.to(device)
            phraseFra = phraseFra.to(device)
            input = phraseEng
            input = input.unsqueeze(1)
            length,batch_size = input.shape #find length of the input, batch size is 1. 
            input = input.to(device)

            #first step # output log probability for SOS sequences: 
            output, hidden = self.EncoderRNN(input)
            decoder_input = torch.zeros((1,batch_size), device=device).long() + vocFra.SOS
  
            hidden = hidden.to(device)
            input_embedded = self.embedder_fr(decoder_input)
            input_embedded = input_embedded.to(device)

            output, hidden_next = self.gru_decode(input_embedded,hidden)
            output = self.decoder(output)
            probas = torch.log(output)
       
            probas,indices = torch.topk(probas.flatten(),k=3,dim=0)
            prob=[i.item() for i in probas]
            indx=[i.item() for i in indices] 
           
       
            for idx,i in enumerate(indx): 
                k_strings[idx].append(i)
                k_strings
            #     k_strings[idx]=[k_strings[idx] for i in range(k)]
            ic(k_strings)
            for idx,i in enumerate(prob):
                k_probas[idx]+=i
            #     k_probas[idx]=[k_probas[idx]]*k
            ic(k_probas)

            #next step: 
            nb_char = 0 
            h0=hidden_next
            
            for step in range(length-1):
                nb_char+=1
                new_prob=[]
                new_indx=[]
                new_k_strings = []
                new_k_probas = []
                char=[]
                P=[]
                for i in range(k):
                 
          
                    input = torch.tensor(indx[i],device=device)
                    input = torch.unsqueeze(input,0)
                    input = torch.unsqueeze(input,1)
  
                    input_embedded = self.embedder_fr(torch.tensor(input)) #o tcheque les 3 valeurs 
                 
                    output,hidden_next = self.gru_decode(input_embedded,h0)
                    output = self.decoder(output)
                    probas = torch.log(output)
                    probas,indices = torch.topk(probas.flatten(),k=3,dim=0)
                    current_prob=[i.item() for i in probas]
                    current_indx=[i.item() for i in indices]    
                    new_prob +=current_prob #9 valeurs, 3 meilleurs proba des 3 choix. 
                    new_indx +=current_indx # 9 valeurs, 3 meilleurs index des 3 choix. 
                    char.append(indices)
                    P.append(k_probas[i] + probas)
                    
                # on va garder les 3 meilleurs proba de new_prob, puis on a l'index des 3 meilleurs proba, et on index la liste des index pour avoir les 3 meilleurs index. 
                char=torch.stack(char,dim=0)
                P = torch.stack(P, dim=0)
               
                best_probas, best_indices = torch.topk(P.flatten()/(nb_char ** 0.8), k, dim=0)
                best_probas = best_probas * (nb_char ** 0.8)
                rows = (best_indices // k).int()  #donne la ligne
                cols = best_indices % k
                for i in range(k):
                    new_k_strings.append(k_strings[rows[i]] + [char[rows[i], cols[i]].item()])
                    new_k_probas.append(best_probas[i].item())
                k_strings = new_k_strings
                k_probas = new_k_probas
                ic(k_strings)
                ic(k_probas)

                

                print('jump to next word. ')

            phraseFra_hat_1=torch.tensor(k_strings[0]) 
            phraseFra_hat_2=torch.tensor(k_strings[1]) 
            phraseFra_hat_3=torch.tensor(k_strings[2]) 
        
   
            print('Phrase anglais:', vocEng.getwords(phraseEng))
            print('Phrase français (truth):', vocFra.getwords(phraseFra))
            print('1ere Traduction:', vocFra.getwords(phraseFra_hat_1))
            print('2eme Traduction:', vocFra.getwords(phraseFra_hat_2))
            print('3eme Traduction:', vocFra.getwords(phraseFra_hat_3))






                


# WITHOUT SEGMENTATION: 
## Instanciate without segmentation: 
NoSegmentation = True # to change to use semgentation or not 
if NoSegmentation : 

    embedding_size = 200
    hidden_size = 200
    embedding_dim = len(vocFra)
    output_size =len(vocFra)
    nb_oov=30
    lr = 0.001

    embedder_ang = torch.nn.Embedding(num_embeddings=embedding_dim,embedding_dim=embedding_size)
    gru_encode = torch.nn.GRU(input_size = embedding_size,hidden_size = hidden_size,num_layers=2,batch_first=False) 
    embedder_fr = torch.nn.Embedding(num_embeddings = embedding_dim,embedding_dim=embedding_size)
    gru_decode = torch.nn.GRU(input_size = embedding_size,hidden_size = hidden_size,num_layers=2,batch_first=False)
    decoder = nn.Linear(in_features=hidden_size, out_features=output_size)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    encode_params = list(embedder_ang.parameters()) + list(gru_encode.parameters())
    decode_params = list(embedder_fr.parameters())  + list(gru_decode.parameters()) + list(decoder.parameters())


            
    #training without segmentation
    training = train_global(criterion,gru_encode,gru_decode,decoder,embedder_fr,embedder_ang,opt='Adam',device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),ckpt_save_path='./models/Traduction')
    loss = training.fit(traindata=train_loader,testdata=test_loader,encode_params=encode_params,decode_params=decode_params,batch_size=BATCH_SIZE,hidden_size=hidden_size,nb_classes=output_size,nb_oov=10,n_epochs=50)
    generation_greedy = training.generate_greedy(testdata=datatest)
    generation_beam = training.generate_last_beam(testdata=datatest,k=3)



#With segmentation : 


##instanciate data: 

else : 
    output_size = len(sp)
    embedding_size = 200
    hidden_size = 200
    embedding_dim = len(sp)  #output size pour la couche linéaire de décodage
    NB_EPOCH = 50

    criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id())

    embeddingNgram = nn.Embedding(num_embeddings=embedding_dim, embedding_dim=embedding_size)

    rnn_encoderNgram = nn.GRU(input_size=embedding_size, hidden_size=embedding_size, num_layers=1 , bias=True)
    rnn_decoderNgram = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=1 , bias=True)
    decoder_layerNgram = nn.Linear(in_features=hidden_size, out_features=output_size)
    encode_params = list(embeddingNgram.parameters()) + list(rnn_encoderNgram.parameters())
    decode_params = list(embeddingNgram.parameters())  + list(rnn_decoderNgram.parameters()) + list(decoder_layerNgram.parameters())


    training = train_global(criterion,rnn_encoderNgram,rnn_decoderNgram,decoder_layerNgram,embeddingNgram,embeddingNgram,opt='Adam',device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),ckpt_save_path='./models/Traduction/Encoded')
    loss = training.fit(traindata=train_loader_ngram,testdata=test_loader_ngram,encode_params=encode_params,decode_params=decode_params,batch_size=BATCH_SIZE,hidden_size=hidden_size,nb_classes=output_size,nb_oov=10,n_epochs=3)

    generation_greedy = training.generate_greedy(testdata=datatest_ngram)

    generation_beam = training.generate_last_beam(testdata=datatest_ngram,k=3)

