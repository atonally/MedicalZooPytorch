import os, csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import lib.utils as utils

organ_dicts = {1: "Liver", 2: "Spleen", 3: "Lungs", 4:"Thyroid", 5:"Kidneys", 6:"Pancreas",
               7: "Gallbladder", 8: "Bladder", 9: "Aorta", 10: "Trachea", 11: "Sternum", 
               16: "Colon", 17: "Stomach", 18: "Heart", 19:"Esophagus", 23: "Bone", 25:"PatientMask"}


class TensorboardWriter():

    def __init__(self, args):

        name_model = args.save + args.model + "_" + args.training_type + "_" + utils.datestr()
        self.writer = SummaryWriter(log_dir=args.save + name_model, comment=name_model)

        self.save = args.save
        self.csv_train, self.csv_val = self.create_stats_files(args.save)
        self.dataset_name = args.training_type
        self.classes = args.classes
        self.label_names = ["Background"]
        if args.training_type == 'CT_organSeg':
            self.label_names.extend([organ_dicts[oo] for oo in args.organs])
        else: 
            self.label_names.extend([str(xx) for xx in range(1, args.classes)])

        self.data = self.create_data_structure()
        
    def create_data_structure(self, ):
        data = {"train": dict((label, 0.0) for label in self.label_names),
                "val": dict((label, 0.0) for label in self.label_names)}
        data['train']['loss'] = 0.0
        data['val']['loss'] = 0.0
        data['train']['count'] = 1.0
        data['val']['count'] = 1.0
        data['train']['dsc'] = 0.0
        data['val']['dsc'] = 0.0
        return data

    def display_terminal(self, iter, epoch, mode='train', summary=False):
        """

        :param iter: iteration or partial epoch
        :param epoch: epoch of training
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        :param summary: to print total statistics at the end of epoch
        """
        if summary:
            info_print = "\nSummary {} Epoch {:2d}:  Loss:{:.4f} \t DSC:{:.4f}  ".format(mode, epoch,
                                                                                         self.data[mode]['loss'] ,
                                                                                         self.data[mode]['dsc'] )

            for i in range(len(self.label_names)):
                info_print += "\t{} : {:.4f}".format(self.label_names[i],
                                                     self.data[mode][self.label_names[i]] )

            print(info_print)
        else:

            info_print = "\nEpoch: {:.2f} Loss:{:.4f} \t DSC:{:.4f}".format(iter, self.data[mode]['loss'],
                                                                            self.data[mode]['dsc'])
            for i in range(len(self.label_names)):
                info_print += "\t{}:{:.4f}".format(self.label_names[i],
                                                   self.data[mode][self.label_names[i]] )
            print(info_print)

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, 'train.csv'), 'w+')
        val_f = open(os.path.join(path, 'val.csv'), 'w+')
        return train_f, val_f

    def reset(self, mode):
        self.data[mode]['dsc'] = 0.0
        self.data[mode]['loss'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0

    def update_scores(self, iter, loss, channel_score, mode, writer_step):
        """
        :param iter: iteration or partial epoch
        :param loss: any loss torch.tensor.item()
        :param channel_score: per channel score or dice coef
        :param mode: train or val ( for training and validation)
        :param writer_step: tensorboard writer step
        """
        # WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY

        dice_coeff = np.mean(channel_score) * 100

        num_channels = len(channel_score)
        self.data[mode]['dsc'] = dice_coeff
        self.data[mode]['loss'] = loss
        self.data[mode]['count'] = iter + 1

        for i in range(num_channels):
            self.data[mode][self.label_names[i]] = channel_score[i]
            if self.writer is not None:
                self.writer.add_scalar(mode + '/' + self.label_names[i], channel_score[i], global_step=writer_step)

    def write_end_of_epoch(self, epoch):

        self.writer.add_scalars('DSC/', {'train': self.data['train']['dsc'],
                                         'val': self.data['val']['dsc'],
                                         }, epoch)
        self.writer.add_scalars('Loss/', {'train': self.data['train']['loss'],
                                          'val': self.data['val']['loss'],
                                          }, epoch)
        for i in range(len(self.label_names)):
            self.writer.add_scalars(self.label_names[i],
                                    {'train': self.data['train'][self.label_names[i]] ,
                                     'val': self.data['val'][self.label_names[i]],
                                     }, epoch)

        train_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}'.format(epoch,
                                                                     self.data['train']['loss'],
                                                                     self.data['train']['dsc'] )
        
        for i in range(len(self.label_names)):
            train_csv_line = train_csv_line + ' ' + self.label_names[i] + ':' + str(self.data['train'][self.label_names[i]])
        
        val_csv_line = 'Epoch:{:2d} Loss:{:.4f} DSC:{:.4f}'.format(epoch,
                                                                   self.data['val']['loss'] ,
                                                                   self.data['val']['dsc'] )
        
        for i in range(len(self.label_names)):
            val_csv_line = val_csv_line + ' ' + self.label_names[i] + ':' + str(self.data['val'][self.label_names[i]])
        
        self.csv_train.write(train_csv_line + '\n')
        self.csv_val.write(val_csv_line + '\n')
      
