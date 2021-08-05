import numpy as np
import torch

from medzoopytorch.utils import prepare_input
from medzoopytorch.test.BaseWriter import TensorboardWriter


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.savedir = args.save
        self.save_frequency = 1
        self.terminal_show_freq = 10 #self.args.terminal_show_freq
        
        if self.args.resume is None:
            self.start_epoch = 1
        else:
            startinfo = np.load(self.savedir + '/logs.npy')
            self.start_epoch = np.floor(startinfo[-1,0]).astype(np.int)
        
            
    def training(self):
        if self.args.resume is None:
            startinfo = np.zeros([1, self.args.classes+2])
            startinfo_val = np.zeros([1, 2*self.args.classes+3])
            np.save(self.savedir + '/logs', startinfo)
            np.save(self.savedir + '/logs_val', startinfo_val)
        else:
            startinfo = np.load(self.savedir + '/logs.npy')
            startinfo_val = np.load(self.savedir + '/logs_val.npy')
        for epoch in range(self.start_epoch, self.args.nEpochs):
            
            if self.do_validation:
                self.validate_epoch(epoch)
            self.train_epoch(epoch)
            
             
            val_loss = self.writer.data['val']['loss'] 
            
            if self.args.save is not None and (epoch % self.save_frequency)==0:
                self.model.save_checkpoint(self.args.save,
                                           epoch, val_loss,
                                           optimizer=self.optimizer)
            self.lr_scheduler.step()

            self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()

            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            if self.args.model=='DEEPMEDIC':
                input_tensor[0].requires_grad=True
                input_tensor[1].requires_grad=True
            else:
                input_tensor.requires_grad = True
            output = self.model(input_tensor)
            loss_dice, per_ch_score = self.criterion(output, target)
            loss_dice.backward()
            self.optimizer.step()
          
        
            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')
                
                oldfile = np.load(self.savedir + '/logs.npy')
                tosave = [partial_epoch, float(loss_dice.cpu().detach().numpy())]
                tosave.extend(per_ch_score.tolist())
                tosave = np.vstack((oldfile, tosave))
                np.save(self.savedir + '/logs.npy', tosave)
        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()
        batch_loss = []
        ch_score = np.zeros((1, self.args.classes))
        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                if self.args.model=='DEEPMEDIC':
                    input_tensor[0].requires_grad=True
                    input_tensor[1].requires_grad=True
                else:
                    input_tensor.requires_grad = True
                output = self.model(input_tensor)
                loss, per_ch_score = self.criterion(output, target)
                batch_loss.append(loss.item())
                ch_score = np.vstack((ch_score, per_ch_score))#ch_score.append(per_ch_score)
        loss = np.mean(batch_loss)
        loss_std = np.std(batch_loss)
        per_ch_score = np.zeros((self.args.classes,))
        per_ch_std = np.zeros((self.args.classes,))
        for cl in range(self.args.classes):
            per_ch_score[cl] = np.mean(ch_score[ch_score[:,cl]>0, cl], axis=0)
            per_ch_std[cl] = np.std(ch_score[ch_score[:,cl]>0, cl], axis=0)
        self.writer.update_scores(batch_idx, loss, per_ch_score, 'val',
                                   epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)
        oldfile = np.load(self.savedir + '/logs_val.npy')
        tosave = [epoch, float(loss)]
        tosave.append(loss_std)      
        tosave.extend(per_ch_score.tolist())
        tosave.extend(per_ch_std.tolist())

        tosave = np.vstack((oldfile, tosave))
        np.save(self.savedir + '/logs_val.npy', tosave)