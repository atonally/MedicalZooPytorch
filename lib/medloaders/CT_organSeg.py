

import glob

import numpy as np
import torch
from torch.utils.data import Dataset

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders.medical_loader_utils import create_sub_volumes

class CT_organSeg(Dataset):


    def __init__(self, args, mode, samples, sample_idx, dataset_path, crop_dim=(32, 32, 32), load=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param fold_id: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        
        self.CLASSES = args.classes
        self.organs = args.organs
        self.full_vol_dim = None 
        self.threshold = 0.1
        self.normalization = "max_min"
        self.augmentation = True
        self.crop_size = crop_dim
        self.list = []
        self.full_volume = None
        self.voxel_resamp = args.voxel_resamp
                    
        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])+ '_' + str(self.voxel_resamp[0]) + 'mm'

        self.save_name = args.outputfolder + '/Data/list-' + mode + '-samples-' + str(
           samples) + '_' + subvol + '_' + str(self.organs) + '.txt'
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), 
                            augment3D.RandomRotation(),
                            augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.7)
        if load:
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            return
        
        CTlist = []
        for directory in args.ct_dir: 
            dir_CTs = glob.glob(directory + "/*.n*")
            pt_labels = [ xx.replace('_', '-') for xx in dir_CTs]
            inds = np.argsort(pt_labels).astype(int)
            dir_CTs = [dir_CTs[ii] for ii in inds]
            CTlist.extend(dir_CTs)

        labellist = []
        for directory in args.gt_dir: 
            dir_labs = glob.glob(directory + "/*.n*")
            pt_labels = [ xx.replace('_', '-') for xx in dir_labs]
            inds = np.argsort(pt_labels).astype(int)
            dir_labs = [dir_labs[ii] for ii in inds]
            labellist.extend(dir_labs)
 
    
        
        if len(CTlist)!=len(labellist):
            print("Error: CT and label list lengths do not match")
        ## NOTE: labellist should be in same order as CT list since patient names are the same, double check this
        
        if self.mode == 'train':
            self.samples = args.samples_train
            
           
            subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])+ '_' + str(self.voxel_resamp[0]) + 'mm'
            self.sub_vol_path = args.outputfolder + '/Data/' + str(args.organs) + mode + subvol + '/'
   
            utils.make_dirs(self.sub_vol_path)

            list_IDsCTs = [CTlist[ii] for ii in sample_idx]
            labels = [labellist[ii] for ii in sample_idx]

            self.list = create_sub_volumes(list_IDsCTs, labels, dataset_name="CT_organSeg",
                                           organs = self.organs,
                                           mode=mode, full_vol_dim=self.full_vol_dim,
                                           crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                           samples = self.samples, normalization=self.normalization, 
                                           resamp_vox = self.voxel_resamp)


        elif self.mode == 'val':
            self.samples = args.samples_val
            
            subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2]) + '_' + str(self.voxel_resamp[0]) + 'mm' 
            self.sub_vol_path = args.outputfolder + '/Data/' + str(args.organs) + mode + subvol + '/'
   
            utils.make_dirs(self.sub_vol_path)
            
            list_IDsCTs = [CTlist[ii] for ii in sample_idx]
            labels = [labellist[ii] for ii in sample_idx]
            
            
            self.list = create_sub_volumes(list_IDsCTs, labels, dataset_name="CT_organSeg",
                                           organs = self.organs,
                                           mode=mode, full_vol_dim=self.full_vol_dim,
                                           crop_size=self.crop_size,
                                           sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                           samples = self.samples, normalization=self.normalization, 
                                           resamp_vox = self.voxel_resamp)

        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        ct_path, seg_path = self.list[index]
        ct,  s = np.load(ct_path), np.load(seg_path)

        if self.mode == 'train' and self.augmentation:
            #print('augmentation reee')
            [augmented_ct], augmented_s = self.transform([ct], s)

            return torch.FloatTensor(augmented_ct.copy()).unsqueeze(0), torch.FloatTensor(augmented_s.copy())
        
        return torch.FloatTensor(ct).unsqueeze(0), torch.FloatTensor(s)
