import glob, random
import os
import numpy as np
from torch.utils.data import DataLoader

import medzoopytorch.utils as utils
from .CT_organSeg import CT_organSeg
from .CT_boneSeg import CT_boneSeg
from .PETCT_lesionDetection import PETCT_lesionDetection


def generate_datasets(args, path):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 0}
    params_val = {'batch_size': args.batchSz_val,
              'shuffle': True,
              'num_workers': 0}
    samples_train = args.samples_train
    samples_val = args.samples_val
    train_val_split = args.train_val_split
    
    #find number of files in CTdir
    CTims = []
    if isinstance(args.ct_dir[0], list):
        for directory in args.ct_dir[0]: 
            CTims.extend(glob.glob(directory + "*.n*"))
    else:
        for directory in args.ct_dir: 
            CTims.extend(glob.glob(directory + "*.n*"))

    CTims.sort()
    
    if args.keepSplit and not args.VFoldCV:
        train_idx = np.load(args.save +'/train_inds_' + str(len(CTims)) + 'patients.npy')
        val_idx =  np.load(args.save +'/val_inds_' + str(len(CTims)) + 'patients.npy')

    elif not args.VFoldCV:
        train_val = random.sample(range(len(CTims)), int(np.ceil(1.0*len(CTims)))) #randomly sample patients for train and val
       
        # split train and val to feed into loader
        train_idx = train_val[0:int(np.ceil(train_val_split*len(CTims)))]
        val_idx = train_val[int(np.ceil(train_val_split*len(CTims))):]

        #randomly sample for test, save indices for later
        np.save(args.save + '/train_inds_' + str(len(CTims)) + 'patients', train_idx)
        np.save(args.save + '/val_inds_' + str(len(CTims)) + 'patients', val_idx)
    else:
        if args.Fold == 1 and not args.keepSplit and 'BLandFU' in args.training_type:
            ## for 5 fold CV, we generate all of the patient inds for the folds but then only train the 1st fold
            ## to continue training the remaining folds, set args.Fold = 2 or more
            # if data contains both baseline and follow-up data, we have to group the baseline patients with the follow-up patients
            CTimsBL = [xx for xx in CTims if 'Baseline' in xx]
            CTimsFU = [xx for xx in CTims if 'Follow-up' in xx]
            
            CTnumsBL = [xx[80:83] for xx in CTimsBL]
            CTnumsFU = [xx[81:84] for xx in CTimsFU]
            CTnums = np.concatenate((CTnumsBL, CTnumsFU), 0)
            folds = [ xx % 5 for xx in range(len(CTims))]
            random.shuffle(folds)
            for xx in range(len(folds)):
                if 'Follow-up' in CTims[xx]:
                    intersect = [kk for kk in range(len(CTnumsBL)) if CTnums[xx] == CTnumsBL[kk]]
                    if len(intersect)>0:
                        folds[xx] = folds[intersect[0]]
                    
            for f in range(5):
                train_val = [xx for xx in range(len(folds)) if folds[xx] != f]
                test_idx = [xx for xx in range(len(folds)) if folds[xx] == f]
                # split train and val to feed into loader
                random.shuffle(train_val)
                train_idx = train_val[0:int(np.ceil(train_val_split*len(train_val)))]
                val_idx = train_val[int(np.ceil(train_val_split*len(train_val))):]
        
                #randomly sample for test, save indices for later
                if not os.path.isdir(args.save + '/Fold' + str(f+1)):
                    utils.make_dirs(args.save + '/Fold' + str(f + 1))
                np.save(args.save + '/Fold' + str(f+1) + '/train_inds_' + str(len(CTims)) + 'patients', train_idx)
                np.save(args.save + '/Fold' + str(f+1) + '/val_inds_' + str(len(CTims)) + 'patients', val_idx)
                np.save(args.save + '/Fold' + str(f+1) + '/test_inds_' + str(len(CTims)) + 'patients', test_idx)
            args.save = args.save + '/Fold1'
            train_idx = np.load(args.save +'/train_inds_' + str(len(CTims)) + 'patients.npy')
            val_idx =  np.load(args.save +'/val_inds_' + str(len(CTims)) + 'patients.npy')
            
        elif args.Fold == 1 and not args.keepSplit:
            ## for 5 fold CV, we generate all of the patient inds for the folds but then only train the 1st fold
            ## to continue training the remaining folds, set args.Fold = 2 or more
            folds = [ xx % 5 for xx in range(len(CTims))]
            random.shuffle(folds)
            for f in range(5):
                train_val = [xx for xx in range(len(folds)) if folds[xx] != f]
                test_idx = [xx for xx in range(len(folds)) if folds[xx] == f]
                # split train and val to feed into loader
                random.shuffle(train_val)
                train_idx = train_val[0:int(np.ceil(train_val_split*len(train_val)))]
                val_idx = train_val[int(np.ceil(train_val_split*len(train_val))):]
        
                #randomly sample for test, save indices for later
                if not os.path.isdir(args.save + '/Fold' + str(f+1)):
                    utils.make_dirs(args.save + '/Fold' + str(f + 1))
                np.save(args.save + '/Fold' + str(f+1) + '/train_inds_' + str(len(CTims)) + 'patients', train_idx)
                np.save(args.save + '/Fold' + str(f+1) + '/val_inds_' + str(len(CTims)) + 'patients', val_idx)
                np.save(args.save + '/Fold' + str(f+1) + '/test_inds_' + str(len(CTims)) + 'patients', test_idx)
            args.save = args.save + '/Fold1'
            train_idx = np.load(args.save +'/train_inds_' + str(len(CTims)) + 'patients.npy')
            val_idx =  np.load(args.save +'/val_inds_' + str(len(CTims)) + 'patients.npy')
        else:
            args.save = args.save + '/Fold' + str(args.Fold)
            train_idx = np.load(args.save +'/train_inds_' + str(len(CTims)) + 'patients.npy')
            val_idx =  np.load(args.save +'/val_inds_' + str(len(CTims)) + 'patients.npy')
        
        
    if args.training_type == "CT_organSeg":
        train_loader = CT_organSeg(args, 'train', samples_train, train_idx, dataset_path=path, crop_dim=args.dim, load=args.loadData)

        val_loader = CT_organSeg(args, 'val', samples_val, val_idx, dataset_path=path, crop_dim=args.dim, load=args.loadData)
        val_generator = DataLoader(val_loader, **params_val)
    if args.training_type == "PETCT_lesionDetection" or args.training_type == 'PETCT_lesionDetection_BLandFU':
        train_loader = PETCT_lesionDetection(args, 'train', samples_train, train_idx, dataset_path=path, crop_dim=args.dim, load=args.loadData)

        val_loader = PETCT_lesionDetection(args, 'val', samples_val, val_idx, dataset_path=path, crop_dim=args.dim, load=args.loadData)
        val_generator = DataLoader(val_loader, **params_val)
    if args.training_type == "CT_boneSeg":
        train_loader = CT_boneSeg(args, 'train', samples_train, train_idx, dataset_path=path, crop_dim=args.dim, load=args.loadData)
        val_loader = CT_boneSeg(args, 'val', samples_val, val_idx, dataset_path=path, crop_dim=args.dim, load=args.loadData)
        #val_generator = None
        val_generator = DataLoader(val_loader, **params_val)

    training_generator = DataLoader(train_loader, **params)


    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator
