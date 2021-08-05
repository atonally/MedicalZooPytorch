# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 17:31:12 2020

@author: amy.weisman
"""


from __future__ import absolute_import, print_function, division
import os

def getAbsPathEvenIfRelativeIsGiven(pathGiven, absolutePathToWhereRelativePathRelatesTo) :
    #os.path.normpath "cleans" Additional ../.// etc.
    if os.path.isabs(pathGiven) : 
        return os.path.normpath(pathGiven)
    else : #relative path given. Need to make absolute path
        if os.path.isdir(absolutePathToWhereRelativePathRelatesTo) :
            relativePathToWhatGiven = absolutePathToWhereRelativePathRelatesTo
        elif os.path.isfile(absolutePathToWhereRelativePathRelatesTo) :
            relativePathToWhatGiven = os.path.dirname(absolutePathToWhereRelativePathRelatesTo)
        else : #not file, not dir, exit.
            print("ERROR: in [func:returnAbsolutePathEvenIfRelativePathIsGiven()] Given path :", absolutePathToWhereRelativePathRelatesTo, " does not correspond to neither an existing file nor a directory. Exiting!"); exit(1)
        return os.path.normpath(relativePathToWhatGiven + "/" + pathGiven)
    
def parseAbsFileLinesInList(pathToListingFile) :
    # os.path.normpath below is to "clean" the paths from ./..//...
    pathToFolderContainingThisListFile = os.path.dirname(pathToListingFile)
    list1 = []
    with open(pathToListingFile, "r") as inp :
        for line in inp :
            if line.strip() == "-" : # symbol indicating the non existence of this channel. Will be zero-filled.
                list1.append("-")
            elif not line.startswith("#") and line.strip() != "" :
                pathToFileParsed = line.strip()
                if os.path.isabs(pathToFileParsed) : #abs path.
                    list1.append(os.path.normpath(pathToFileParsed))
                else : #relative path to this listing-file.
                    list1.append(os.path.normpath(pathToFolderContainingThisListFile + "/" + pathToFileParsed))
    return list1

    
class Config(object):
    
    def __init__(self, abs_path_to_cfg):
        self._configStruct = {}
        self._abs_path_to_cfg = abs_path_to_cfg # for printing later.
        print("Given configuration file: ", self._abs_path_to_cfg)
        exec(open(self._abs_path_to_cfg).read(), self._configStruct)
        self._check_for_deprecated_cfg()
        
    def __getitem__(self, key): # overriding the [] operator.
        return self.get(key)
    
    def get(self, string1) :
        return self._configStruct[string1] if string1 in self._configStruct else None
    
    def get_abs_path_to_cfg(self):
        return self._abs_path_to_cfg
    
    def _check_for_deprecated_cfg(self):
        pass
    
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        pass
 

class GetConfig(Config):
    #================ MODEL PARAMETERS =================
    outputfolder = "outputfolder"
    training_type = "training_type"
    
    model = "model"
    FMsLayerOne = "FMsLayerOne"
    organs = "organs"
    inChannels = "inChannels"
    inModalities = "inModalities"
    
    # =============TRAINING========================
    batchSz = "batchSz"
    batchSz_val = "batchSz_val"
    
    nEpochs = "nEpochs"
    opt = "opt"
    samples_train = "samples_train"
    samples_val = "samples_val"
    dim = "dim"
    voxel_resamp = "voxel_resamp"
    train_val_split = "train_val_split"
    
    
    lr = "lr"
    step_size = "step_size"
    gamma = "gamma"

    CT_dir = "CTdirectory"
    GT_dir = "GTdirectory"
    
    # =============TESTING========================
    batchSz_test = "batchSz_test"
    overlap_stepsize = "overlap_stepsize"


    def __init__(self, abs_path_to_cfg):
        Config.__init__(self, abs_path_to_cfg)

   
def addConfigArguments(args, all_args):
    args.outputfolder = all_args[all_args.outputfolder]
    args.training_type = all_args[all_args.training_type]
    
    args.organs = all_args[all_args.organs]
    args.classes = len(args.organs)+1

    args.inChannels = all_args[all_args.inChannels]
    args.inModalities = all_args[all_args.inModalities]
    
    args.model =all_args[ all_args.model]
    args.FMsLayerOne = all_args[all_args.FMsLayerOne]
    
    
    args.batchSz = all_args[all_args.batchSz]
    args.batchSz_val = all_args[all_args.batchSz_val]
    args.nEpochs = all_args[all_args.nEpochs]
    
    args.lr = all_args[all_args.lr]
    args.step_size = all_args[all_args.step_size]
    args.gamma = all_args[all_args.gamma]
    
    args.opt = all_args[all_args.opt]
    args.dim = all_args[all_args.dim]
    args.voxel_resamp= all_args[all_args.voxel_resamp]
    
    args.train_val_split = all_args[all_args.train_val_split]
    args.samples_train = all_args[all_args.samples_train]
    args.samples_val =all_args[ all_args.samples_val]
    args.ct_dir = all_args[all_args.CT_dir]
    args.gt_dir = all_args[all_args.GT_dir]

    args.augmentation = True
    
    args.overlap_stepsize = all_args[all_args.overlap_stepsize]
    args.batchSz_test = all_args[all_args.batchSz_test]
    return args
   
    
    