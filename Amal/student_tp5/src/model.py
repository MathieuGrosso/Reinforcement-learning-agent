
import os.path
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
import torch.optim as optim
import torch.nn as nn
import datetime

#  TODO: 
def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """

    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    CrossEntropyLoss= torch.nn.CrossEntropyLoss(reduce = False)
    
    output=output.view(output.shape[0]*output.shape[1],output.shape[2])
    target=target.view(target.shape[0]*target.shape[1])

    out=CrossEntropyLoss(output,target)

    mask = torch.where(out==PAD_IX,0,1)
    out*=mask
    out = sum(out)/mask.sum()

    ic(out)
    # out=torch.nn.CrossEntropyLoss(output,target,ignore_index=0,reduce=False)
    return out
    
class training_pytorch(nn.Module):
    def __init__(self,criterion=None,opt=None,mode=None,model=None,ckpt_save_path=None):
        super().__init__()
        self.model=model
        self.optimizer=opt
        self.state={}
        self.ckpt_save_path = ckpt_save_path
        self.criterion=criterion
        self.mode = mode
        ic(self.mode)

    def __train_epoch(self, dataloader):
        epoch_loss = 0
        epoch_acc  = 0
        for idx, data in enumerate(dataloader):
            input,label = data
            if self.mode=="forecast":
                output = self(input.long())
            else: 
                output=self(input)
            if self.mode=='generation':
                label = label.reshape(-1) 
                input = input.reshape(-1,input.shape[-1])
                output= output.reshape(-1,output.shape[-1])
            if  self.mode=='forecast':
                label=label.reshape(label.shape[0],label.shape[1],-1)
                n_correct=0
            else: 
                n_correct = (torch.argmax(output, dim=1) == label).sum().item()
            epoch_acc += n_correct/(input.shape[0])


            loss = self.criterion(output, label,PAD_IX)
            loss.backward()
            epoch_loss += loss.item()
            self.opt.step()
            self.opt.zero_grad()
        return epoch_loss/len(dataloader), epoch_acc/len(dataloader)
    
    def __validate(self, dataloader):
        epoch_loss = 0
        epoch_acc  = 0
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            
            if self.mode=='generation' and self.model =='many to many':
                batch_y = batch_y.reshape(-1) 
            if self.model =='many to many' and self.mode=='forecast':
                batch_y=batch_y.reshape(batch_y.shape[0],batch_y.shape[1],-1)
                # ic(batch_y.shape)
                # print('hello')
            batch_output = self(batch_x)
            if self.model == "many to many":
                n_correct = 0
                if self.mode=='generation':
                    batch_output=batch_output.reshape(-1,batch_output.shape[-1])
            if self.model == "many to many":
                n_correct = 0
                # batch_y=batch_y[:,-2:,:,:]
                # ic(batch_y.shape)
                # ic(batch_output.shape)
                loss= self.criterion(batch_output,batch_y)
            else: 
                n_correct = (torch.argmax(batch_output, dim=1) == batch_y).sum().item()
                loss = self.criterion(batch_output, batch_y)
            epoch_acc += n_correct / batch_x.shape[0]
            epoch_loss += loss.item()
        return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

    def fit(self, dataloader,validation_data=None, n_epochs=100, lr=0.001, verbose=10,ckpt=None):
        if self.optimizer=="SGD":
            self.opt = optim.SGD(self.parameters(),lr=0.01,momentum=0.9)
        start_epoch = 0
        total_train_loss=0.0
        total_train_acc=0.0
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
                g['lr'] = state['lr']
        
        for epoch in range(start_epoch,n_epochs): 
            train_loss, train_acc = self.__train_epoch(dataloader)
            total_train_loss+=train_loss
            total_train_acc+=total_train_acc
            print('Epoch {:2d} loss: {:1.4f}  Train acc: {:1.4f} '.format(epoch, train_loss, train_acc))
            if validation_data is not None:
                with torch.no_grad():
                    val_loss, val_acc = self.__validate(validation_data)
                print('Epoch {:2d} loss_val: {:1.4f}  val_acc: {:1.4f} '.format(epoch, val_loss, val_acc))
        
            if self.ckpt_save_path:
                self.state['lr'] = lr
                self.state['epoch'] = epoch
                self.state['state_dict'] = self.state_dict()
                if not os.path.exists(self.ckpt_save_path):
                    os.mkdir(self.ckpt_save_path)
                torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{start_time}_epoch{epoch}.ckpt'))


# class RNN(nn.Module):
#     #  TODO:  Recopier l'implémentation du RNN (TP 4)

class RNN(nn.Module): 
    """many to many RNN"""
    def __init__(self,in_features,out_features,hidden_size,model='many to many',mode=None,save=False):
        super(RNN, self).__init__()
        self.in_features    = in_features
        self.out_features   = out_features
        self.model= model 
        self.save = save
        self.mode = mode
        self.hidden_size    = hidden_size
        self.linear_one_step=nn.Linear(self.hidden_size+self.hidden_size, self.hidden_size, bias=True)
        if self.mode=='generation':
            self.linear_one_step=nn.Linear(self.hidden_size+self.hidden_size, self.hidden_size, bias=True)
        if self.mode == 'forecast':
            self.linear_one_step=nn.Linear(160+self.hidden_size,self.hidden_size,bias=True) #pour l'instant on considére 20 stations donc on met 20, a generaliser autrement. 
        if self.mode =='classification':
            self.linear_one_step=nn.Linear(self.hidden_size + 2, self.hidden_size, bias=True)
 

        self.linear_decode = nn.Linear(self.hidden_size,self.out_features)
        self.embedding  = nn.Embedding(num_embeddings=self.out_features, embedding_dim = self.hidden_size, padding_idx=0)
        self.activation = torch.nn.Tanh()
        self.mode = mode
    
    def one_step(self,x,h):
        h=torch.cat((x,h),dim=1)
        h=self.linear_one_step(h)
        h=self.activation(h)
        return h

    def decode(self,h):
        h = (self.linear_decode(h))
        return h 
    
    def forward(self, x,h=None):
        # x is of shape (B, L, input_dim), we need to extract xt
        
        x = self.embedding(x)
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_size)
        h_seq=[]
        for t in range(x.shape[1]):
            xt = x[:, t, :].reshape(x.shape[0], -1)
            h=self.one_step(xt,h)
            if self.model=='many to many':
                h_seq.append(self.decode(h).reshape(-1, 1, self.out_features))
        if self.model=="many to many":
            output = torch.cat(h_seq,dim=1)
        else: 
            output=self.decode(h)
        return output,h

    

class LSTM(nn.Module):
    def __init__(self, dim_in, dim_out, dim_latent=10):
        super(LSTM, self).__init__()
        self.LinearF = torch.nn.Linear(in_features=dim_latent+dim_in, out_features=dim_latent)  # [h(t-1), x(t)] --> f(t) 
        self.LinearI = torch.nn.Linear(in_features=dim_latent+dim_in, out_features=dim_latent)  # [h(t-1), x(t)] --> i(t) 

        self.LinearC = torch.nn.Linear(in_features=dim_latent+dim_in, out_features=dim_latent)  # [h(t-1), x(t)] --> C(t)

        self.LinearO = torch.nn.Linear(in_features=dim_latent+dim_in, out_features=dim_latent)  # [h(t-1), x(t)] --> o(t)
        
        self.LinearDecode = torch.nn.Linear(in_features=dim_latent, out_features=dim_out)  # h(t) --> y(t)
        self.latent = dim_latent  #dimension d'un etat caché h

    def one_step(self, x, h, C):  #x(t) (batch x dim) --> l'instant t d'un batch de sequence (dimension length disparait); h(t-1) (batch x latent); renvoie h(t) de taille (batch x latent)
        input = torch.cat((h, x), dim=1)
        outf = F.sigmoid(self.LinearF(input))
        outi = F.sigmoid(self.LinearI(input))
        C_new = outf * C + outi * F.tanh(self.LinearC(input))
        outo = F.sigmoid(self.LinearO(input))
        h_new = outo * F.tanh(C_new)
        return h_new, C_new, outf, outi, outo
        
    # Ajout de l'état interne C
    def forward(self, X, h0=None, C0=None, device=torch.device('cpu')):  #X (length x batch x dim), h (batch x latent); renvoie H de taille (length x batch x latent)
        # appel one_step pour chaque ligne x de X (donc on l'appelle length fois)
        length = X.shape[0]
        batch_size = X.shape[1]
        h0 = torch.zeros(batch_size, self.latent).to(device)
        C0 = torch.zeros(batch_size, self.latent).to(device)
        H = [h0]
        C = [C0]
        F, I, O = [], [], []
        for l in range(length): #checker les index ici
            # print('X device', X[l].device)
            # print('H device', H[l].device)
            # print('C device', C[l].device)
            h_new, C_new, outf, outi, outo = self.one_step(X[l], H[l], C[l])
            H.append(h_new)
            C.append(C_new)
            F.append(outf)
            I.append(outi)
            O.append(outo)
        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        F = torch.stack(F, dim=0)
        I = torch.stack(I, dim=0)
        O = torch.stack(O, dim=0)
        return H, C, F, I, O

    def decode(self, h):  #h(t) (batch x latent); renvoie y(t) (batch x output)
        y = self.LinearDecode(h)
        return y


class GRU(nn.Module):
    def __init__(self, dim_in, dim_out, dim_latent=10):
        super(GRU, self).__init__()
        self.LinearZ = torch.nn.Linear(in_features=dim_latent+dim_in, out_features=dim_latent)  # [h(t-1), x(t)] --> z(t) 
        self.LinearR = torch.nn.Linear(in_features=dim_latent+dim_in, out_features=dim_latent)  # [h(t-1), x(t)] --> r(t) 

        self.LinearH = torch.nn.Linear(in_features=dim_latent+dim_in, out_features=dim_latent)  # [r(t) x h(t-1), x(t)] --> h(t)
        
        self.LinearDecode = torch.nn.Linear(in_features=dim_latent, out_features=dim_out)  # h(t) --> y(t)
        self.latent = dim_latent  #dimension d'un etat caché h

    def one_step(self, x, h):  #x(t) (batch x dim) --> l'instant t d'un batch de sequence (dimension length disparait); h(t-1) (batch x latent); renvoie h(t) de taille (batch x latent)
        input = torch.cat((h, x), dim=1)
        outz = F.sigmoid(self.LinearZ(input))
        outr = F.sigmoid(self.LinearR(input))
        h = (1 - outz) * h + outz * F.tanh(self.LinearH(torch.cat((outr * h, x), dim=1)))
        return h, outz, outr
        # return h
        
    # Inchangé par rapport au RNN du TME4
    def forward(self, X, h=None, device=torch.device('cpu')):  #X (length x batch x dim), h (batch x latent); renvoie H de taille (length x batch x latent)
        # appel one_step pour chaque ligne x de X (donc on l'appelle length fois)
        length = X.shape[0]
        batch_size = X.shape[1]
        h0 = torch.zeros(batch_size, self.latent).to(device)
        H = [h0]
        Z = []
        R = []
        for l in range(length):
            h, outz, outr = self.one_step(X[l], H[l])
            H.append(h)
            Z.append(outz)
            R.append(outr)
            # H.append(self.one_step(X[l], H[l]))
        H = torch.stack(H, dim=0)
        Z = torch.stack(Z, dim=0)
        R = torch.stack(R, dim=0)
        return H, Z, R

    def decode(self, h):  #h(t) (batch x latent); renvoie y(t) (batch x output)
        y = self.LinearDecode(h)
        return y


class GRU2(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self,in_features,out_features,hidden_size,model,save):
        super(GRU,self).__init__()
        self.in_features    = in_features
        self.out_features   = out_features
        self.hidden_size    = hidden_size
        self.model = model
        self.save = save
        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()
        self.linearZ = (nn.Linear(in_features= self.hidden_size+self.hidden_size,out_features= self.hidden_size,bias=False))
        self.linearR = (nn.Linear(in_features= self.hidden_size+self.hidden_size,out_features=self.hidden_size,bias=False))
        self.linearH_2 = (nn.Linear(in_features= self.hidden_size+self.hidden_size,out_features=self.hidden_size,bias=False))

        self.LinearDecode = nn.Linear(self.hidden_size,self.out_features)

        self.embedding = nn.Embedding(num_embeddings=self.out_features, embedding_dim = self.hidden_size)


    def make_embedding(self,x):
        x =self.embedding(x)
        return x


    def one_step(self,x,ht):

        cat = torch.cat((ht,x),dim=1)

        zt = self.sigmoid(self.linearZ(cat))
        rt = self.sigmoid(self.linearR(cat))
        cat_2 = torch.cat((rt*ht,x),dim=1)
        ht_tilde = self.tanh(self.linearH_2(cat_2))
        h = (1-zt)*ht+zt*ht_tilde
        return h,zt,rt



    def decode(self,h):
        return self.LinearDecode(h)



    def forward(self,x,h=None):
        x = self.make_embedding(x)
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_size)
        h_seq=[]
        for t in range(x.shape[1]):
            xt = x[:, t, :].reshape(x.shape[0], -1)
            h,zt,rt=self.one_step(xt,h)
            if self.model=='many to many':
                h_seq.append(self.decode(h).reshape(-1, 1, self.out_features))
        if self.model=="many to many":
            output = torch.cat(h_seq,dim=1)
        else: 
            output=self.decode(h)
        return output,h



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
if __name__ == "__main__":
    start_epoch=0
    nb_epochs=4300
    batch_size=32
    path = '/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/AMAL-student_tp4.2021/data/trump_full_speech.txt'
    txt = open(path)
    train_dataset = TextDataset(path)
    train_dataloader = DataLoader(train_dataset,collate_fn=pad_collate_fn, shuffle=True, batch_size=batch_size, drop_last=False)
    ic(next(iter(train_dataloader)))
    total_train_loss = 0.0
    ckpt='/Users/mathieugrosso/Desktop/Master_DAC/MasterDAC/AMAL/TP/student_tp5/src/models/GRU/ckpt_20211029-124452_epoch4268.ckpt'
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_save_path = './models/GRU'
    ckpt_save_path = './models/RNN'
    state={}
    lr = 0.01
    model = RNN(in_features=1,hidden_size=20,out_features=len(id2lettre),model='many to many',save = False)
    # model = GRU(in_features=1,hidden_size=20,out_features=len(id2lettre),model='many to many',save = True)
    opt = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    if model.save :
        if os.path.isfile(ckpt): 
            print('we load')
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            ic(start_epoch)
            model.load_state_dict(state['state_dict'])
            for g in opt.param_groups:
                g['lr'] = state['lr']




    for epoch in range(start_epoch,nb_epochs): 
        epoch_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            batch=batch.reshape(-1,1)
            input = batch[:-1]
            ic(input.shape)
            label = batch[1:]

            label = label.type(torch.LongTensor)
            
                
            
            output,_ = model(input.long())
            loss = maskedCrossEntropy(output,label,PAD_IX)
            loss.backward()
            epoch_loss += loss.item()
            opt.step()
            opt.zero_grad()
            loss_train = epoch_loss/len(train_dataloader)
                

        total_train_loss += loss_train
        print('Epoch {:2d} loss: {:1.4f}  '.format(epoch, loss_train))
        if model.save:
                
            state['lr'] = lr
            state['epoch'] = epoch
            state['state_dict'] = model.state_dict()
            if not os.path.exists(ckpt_save_path):
                os.mkdir(ckpt_save_path)
            torch.save(state, os.path.join(ckpt_save_path, f'ckpt_{start_time}_epoch{epoch}.ckpt'))


        
    # generation: 
    generate(model, model.embedding,model.decode,eos=EOS_IX)




