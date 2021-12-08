
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from icecream import ic
import torch.optim as optim
import datetime
import os 



class training_pytorch(nn.Module):
    def __init__(self,criterion,opt,mode=None,model=None,ckpt_save_path=None):
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


            loss = self.criterion(output, label)
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


class RNN(training_pytorch): 
    """many to many RNN"""
    def __init__(self,in_features,out_features,hidden_size,criterion,opt,model='many to many',ckpt_save_path=None,mode=None):
        super(RNN, self).__init__(criterion=criterion,  opt=opt,model=model,ckpt_save_path=ckpt_save_path,mode=mode)
        self.in_features    = in_features
        self.out_features   = out_features
        self.hidden_size    = hidden_size
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
        if self.model=="generation":
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
        return output

    def predict(self,x,sequence_length):
        """
        Args:
            x: Input int tensor of shape (1)
        Returns:
            y_seq: Sequence of ints of length seq_length
        """
        #initialisation: 
        y_seq = []
        ic(x.shape)
       
        x=self.embedding(x.long())
        x=x.reshape(x.shape[0],-1)
        ic(x.shape)
        h=torch.zeros(x.shape[0],self.hidden_size)
        ic(h.shape)
        # h=torch.cat((x,h),dim=1)
        h = self.one_step(x,h)
        yhat = self.decode(h)

        yhat = torch.argmax(torch.softmax(yhat,dim=-1),dim=-1)
        y_seq.append(yhat)
        yhat=self.embedding(yhat)

        #generate other terms: 
        for t in range(sequence_length-1):
            h = self.one_step(yhat, h)
            yhat = torch.softmax(self.decode(h),dim=-1)
            yhat = torch.argmax(yhat, -1)
            y_seq.append(yhat)
            yhat = self.embedding(yhat)

 
        return yhat









        





