from medzoopytorch.medloaders import medical_image_process as img_loader
import numpy as np
import torch

def get_viz_set(*ls, dataset_name, organs, resamp_vox, label = True, save=False, sub_vol_path=None, toCanonical = True, fillEmpty = False):
    """
    Returns total 3d input volumes (t1 and t2 or more) and segmentation maps
    3d total vol shape : torch.Size([1, 144, 192, 256])
    
    This function is technically only used during testing, could move but don't want to break anything - Amy
    """
    total_volumes = []
    if label:
        label_path= ls[-1]
        ls = ls[0:len(ls)-1]
 
        
    full_img_tensor, img_info, orig_imsize = img_loader.load_medical_image(ls[0], type="CT", reference_image = None, viz3d = False, resample = resamp_vox, normalization='max_min',
                                                                             rescale = None, crop = None, crop_size = None, to_canonical = toCanonical, fillEmpty = fillEmpty)
    
    total_volumes.append(full_img_tensor.to(torch.float32))
    if len(ls)>1:
        for path in ls[1:]:
            full_img_tensor, _, _ = img_loader.load_medical_image(path, type="PET", reference_image = ls[0], viz3d = False, resample = resamp_vox, normalization='max_min',
                                                                                 rescale = None, crop = None, crop_size = None, to_canonical = toCanonical, fillEmpty = fillEmpty)
        
            total_volumes.append(full_img_tensor.to(torch.float32))
        
    if label:
        full_segmentation_map, _, _= img_loader.load_medical_image(label_path, type="PET", reference_image = ls[0], viz3d = False, resample = None, normalization='max_min',
                                                                               rescale = None, crop = None, crop_size = None, to_canonical = toCanonical)
    else:
        full_segmentation_map = []
    
    
    return ls[0], img_info, orig_imsize, torch.stack(total_volumes, dim=0), full_segmentation_map




def create_sub_volumes(*ls, dataset_name, mode, organs, full_vol_dim, crop_size, sub_vol_path, normalization='full_volume_mean',
                       samples, th_percent=0.0, resamp_vox = [1.2, 1.2, 1.2], model = 'UNET'):
    """

    :param ls: list of modality paths, where the last path is the segmentation map
    :param dataset_name: which dataset is used
    :param mode: train/val
    :param samples: train/val samples to generate
    :param full_vol_dim: full image size
    :param crop_size: train volume size
    :param sub_vol_path: path for the particular patient
    :param th_percent: the % of the croped dim that corresponds to non-zero labels
    :param crop_type:
    :returns nothing, saves images 
    """
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    list = []

    organs_new = (0,) + organs
    samples_per_organ = round(samples/(len(organs_new)))
    
        
    removed_ids = []

    print('Mode: ' + mode + ' Subvolume samples to generate per patient: ', samples, 'Volumes: ', total)
    for i in range(total):
        sample_paths = []
        tensor_images = []
        for j in range(modalities):
            sample_paths.append(ls[j][i])
        label_path = sample_paths[-1]
        
        # Load images and corresponding labels 
        full_img_tensor, affine_img, orig_imsize = img_loader.load_medical_image(sample_paths[0], type="CT", reference_image = None, viz3d = False, resample = np.asarray(resamp_vox), normalization=normalization,
                                                                    rescale = None, crop = None, crop_size = None)
        full_vol_dim = np.asarray(full_img_tensor.shape  ) 
                
        all_ims = np.zeros((modalities-1, full_vol_dim[0], full_vol_dim[1], full_vol_dim[2]))
        all_ims[0,:,:,:] = full_img_tensor
        for modal in range(1, modalities-1):
            full_img_tensor, affine_img, orig_imsize = img_loader.load_medical_image(sample_paths[modal], type="PET", reference_image = sample_paths[0], viz3d = False, resample = np.asarray(resamp_vox), normalization=normalization,
                                                                    rescale = None, crop = None, crop_size = None)
            all_ims[modal,:,:,:] = full_img_tensor
        full_img_tensor = torch.from_numpy(all_ims.copy())
                                          
        full_segmentation_map, affine_seg, orig_imsize = img_loader.load_medical_image(label_path,  type='label', reference_image = sample_paths[0], viz3d=False, resample = np.asarray(resamp_vox), rescale = None,
                                                                          crop = None, crop_size = None)
        
        # Sometimes, the image may be smaller than the patch size (128x128x128). If this is the case, pad to patch size
        if np.any(np.divide(full_vol_dim,crop_size)<1):
            full_img_tensor, full_segmentation_map, full_vol_dim = pad_to_patchsize(full_img_tensor, full_segmentation_map, crop_size)

        # map the dataset organs to the correct U-net standard organ labels, then remap back to being just 0, 1, 2, 3, 4, ...
        if dataset_name=='PETCT_lesionDetection':
            full_segmentation_map = full_segmentation_map>0
        elif dataset_name == 'CT_boneSeg':
            full_segmentation_map = fix_bone_map(full_segmentation_map, organs)
        else:
            full_segmentation_map = fix_seg_map(full_segmentation_map, organs, label_path)
        if model=='DEEPMEDIC':
            full_img_tensor, full_segmentation_map, full_vol_dim = pad_bygiven_size(full_img_tensor, full_segmentation_map, 128)

            
            
            
        full_segmentation_map = full_segmentation_map.clone().to(torch.int16)
        
        # REMOVE THE SUBJECT FROM THE DATASET IF ALL OF THE ORGANS ARE NOT PRESENT
        #if len(np.unique(full_segmentation_map)) == len(organs_new):
        # iterate through organs 
        for org in range(len(organs_new)):
            samples_complete=0
            # I require that there are a certain number of samples per organ, so I just iterate through a lot of samples until I hit that goal, then break once it's hit
            for k in range(50*samples_per_organ):    
                if samples_complete<samples_per_organ:
                    # find a random patch of the right patch size 
                    crop = find_random_crop_dim(full_vol_dim, crop_size)
                    segmentation_map = img_loader.crop_img(full_segmentation_map, crop_size, crop)
                    
                    
                    tensor_images = [] # having the tensor images be a list is used in case there are multiple channels/modalities 
                    
                    # we want fully empty patches for background class, to make sure we are training on empty patches (e.g., brain)
                    if org==0:
                        criterion = sum(sum(sum(segmentation_map!=org)))==0
                    else: 
                        if model == 'DEEPMEDIC':
                            criterion = sum(sum(sum(segmentation_map[42:87, 42:87, 42:87]==org)))>0 
                        else:
                            criterion = sum(sum(sum(segmentation_map==org)))>0 
                    
                    if criterion:
                        samples_complete=samples_complete+1
                        
                        for j in range(modalities - 1):
                            img_tensor =  img_loader.crop_img(full_img_tensor[j,:,:,:], crop_size, crop)
                            tensor_images.append(img_tensor)
            
                        filename = sub_vol_path + 'pt_' + str(i) + '_organ' + str(organs_new[org]) + '_sample_' + str(k) + '_modality_'
                        list_saved_paths = []
                        for j in range(modalities - 1):
                            f_t1 = filename + str(j) + '.npy'
                            list_saved_paths.append(f_t1)
                
                            np.save(f_t1, tensor_images[j])
                
                        f_seg = filename + 'seg.npy'
                
                        np.save(f_seg, segmentation_map)
                        list_saved_paths.append(f_seg)
                        list.append(tuple(list_saved_paths))
                      
        #else:
        #    removed_ids.append(i)
    print("REMOVED DATA FROM " + str(len(removed_ids)) + " PATIENT(S) WITH INCORRECT ORGAN LABELS")
    return list



def fix_seg_map(segmentation_map, organs, dataset, map_to_new = True):
    """
    key for mapping the dataset labels to the correct organ labels
    first value: organ label in original dataset
    second value: organ label standard for U-net training
    map_to_new =set to true if you want to re-map to 0, 1, 2, 3, 4, 5... 
    """
    if 'full_labels' in dataset: 
        key = [[1, 1], 
               [2, 2], 
               [3, 3], 
               [4, 4],
               [5, 5],
               [6, 6],
               [8, 8], 
               [9, 9],
               [16, 16], 
               [17, 17],
               [18, 18], 
               [23, 23]]
    elif 'CapeStart_(4, 6, 9, 17)' in dataset:
        key = [[4, 4], 
               [6, 6,],
               [9, 9],
               [17,17]]
    elif 'Skeleton_Luciano' in dataset:
        key = [[1, 23]]
    elif 'Thyroid1' in dataset:
        key = [[1, 4]]
    elif 'remove_table' in dataset: 
        key = [[1, 25]] 
    elif 'CT-org' in dataset:
        key = [[ 1, 1], #...%liver
                [2, 8], #... %bladder
                [3, 3],#... %lung
                [4, 5],#...% kidney
                [5, 23],#...  %bone
                [6, 24]]#... %brain
    elif 'Karmanos' in dataset:
        key = [[1, 23]]#...%bone
    elif 'BTCV' in dataset:
        key = [[6, 1],# ...%liver
                [1, 2],#... %spleen
                [14, 3],#... % right lung
                [14, 3],#... %left lung
                [2, 5],#... %kidney right
                [3, 5],#... %kidney left
                [11, 6],#... %pancreas
                [4, 7],#...%gallbladder
                [8, 9],#... %aorta
                [12, 13],#... %R adrenal
                [13, 13],#... %L adrenal
                [7, 17],#... %stomach
                [5, 19],#... %esophagus
                [9, 21],#... % inf vena cava
                [10, 22]]# % portal and splenic vein
    elif 'Task03' in dataset: 
        key = [[1, 1,], 
               [2, 1]] #liver
    elif 'Task07' in dataset: 
        key = [[1, 6,], 
               [2, 6]] #liver
    elif 'LCTSC' in dataset:
        key = [[1, 3]] #left and right lungs segmented separately
    elif 'MVI' in dataset or 'NCI' in dataset:
        key = [[1, 23]] #left and right lungs segmented separately
    elif 'TCIA_ACRIN_FDG_NSCLC/Baseline/FinalContours' in dataset:
        key = [[1, 1], 
               [2, 2], 
               [3, 3],
               [5, 5], 
               [8, 8],
               [18,18], 
               [23, 23]]
    elif 'TCIA_ACRIN_FDG_NSCLC/Baseline/Colon1_Song/' in dataset:
        key = [[1, 16]]
    elif 'ACRIN-FLT-Breast/CapeStart_bowels' in dataset or 'MDV/Baseline/CapeStartBowel' in dataset:
        key = [[1, 16]] #colon
    elif 'TCIA_ACRIN_FDG_NSCLC' in dataset or 'ACRIN-FLT-Breast' in dataset: 
        key = [[1, 1,], #liver
               [2, 2]] #spleen
    elif 'Task09' in dataset: 
        key = [[1, 2,], 
               [2, 2],
               [3, 2]] #spleen
    elif 'Dotatate/DOTATATE_baseline/colons/' in dataset: 
        key = [[1, 16]] # colon
    elif 'VISCERAL' in dataset:
        key = [[1, 1],# ...%liver
                [2, 2],#... %spleen
                [3, 3],#... % right lung
                [4, 4],#... %thyroid
                [5, 5],#... %kidney right
                [5, 5],#... %kidney left
                [6, 6],#... %pancreas
                [7, 7],#...%gallbladder
                [8, 8],#...%bladder
                [9, 9],#... %aorta
                [10, 10],#... %trachea
                [11, 11],#... %sternum
                [12, 12],#... %L1 vertebra
                [13, 13],#... %R adrenal
                [13, 13],#... %L adrenal
                [14, 14],#... %R psoas major
                [14, 14],#... %L psoas major
                [15, 15],#... %R Rectus
                [15, 15],#... %L Rectus
                [16, 16],#... %bowel
                [17, 17],#... %stomac
                [18, 18],#... heart
                [19, 19],#... %esophagus
                [20, 20],#... %duodenum
                [21, 21],#... % inf vena cava
                [22, 22],# % portal and splenic vein
                [23, 23]] #bone
    elif 'TCIA_pancreas_labels-02-05-2017' in dataset:
        key = [[1,6]] # pancreas only
    elif 'pancreasCT' in dataset:
        key = [[ 6, 1],# ...%liver
                [1, 2],#... %spleen
                [2, 5],#...%left kidney
                [3, 5],#... %kidney right
                [11, 6],#... %pancreas
                [4, 7],#...%gallbladder
                [7, 17],#... %stomach
                [5, 19],#... %esophagus
                [14,20]]# %duodenum
    else: 
        print("dataset not found!")
                
    
    new_segmentation_map = np.zeros(segmentation_map.shape)
    for lab in range(len(key)):
        new_segmentation_map[segmentation_map == key[lab][0]] = key[lab][1] 
    final_segmentation_map = np.zeros(segmentation_map.shape)
    if map_to_new: 
        for lab in range(len(organs)):
            final_segmentation_map[new_segmentation_map == organs[lab]] = lab+1
    else: 
        for lab in range(len(organs)):
            final_segmentation_map[new_segmentation_map == organs[lab]] = organs[lab]
    return torch.from_numpy(final_segmentation_map)

def fix_bone_map(segmentation_map, bones, map_to_new = True):
    """
    key for mapping the dataset labels to the correct organ labels
    first value: organ label in original dataset
    second value: organ label standard for U-net training
    map_to_new =set to true if you want to re-map to 0, 1, 2, 3, 4, 5... 
    """
    new_segmentation_map = segmentation_map
    final_segmentation_map = np.zeros(segmentation_map.shape)
    if map_to_new: 
        for lab in range(len(bones)):
            final_segmentation_map[new_segmentation_map == bones[lab]] = lab+1
    else: 
        for lab in range(len(bones)):
            final_segmentation_map[new_segmentation_map == bones[lab]] = bones[lab]
    return torch.from_numpy(final_segmentation_map)   

def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)

def pad_to_patchsize(full_img_tensor, full_segmentation_map, crop_size):
    new_imsize = [max([crop_size[0], full_img_tensor.shape[1]]),
                  max([crop_size[1], full_img_tensor.shape[2]]),
                  max([crop_size[2], full_img_tensor.shape[3]])]
    
    new_full_img_tensor = torch.zeros([full_img_tensor.shape[0], 
                                       new_imsize[0], new_imsize[1], new_imsize[2]],
                                      dtype = full_img_tensor.dtype)     
    new_full_segmentation_map = torch.zeros([new_imsize[0], new_imsize[1], new_imsize[2]],
                                            dtype = torch.int16)
            
    new_full_img_tensor[:, 0:full_img_tensor.shape[1], 
                        0:full_img_tensor.shape[2], 
                        0:full_img_tensor.shape[3]] = full_img_tensor
    new_full_segmentation_map[0:full_segmentation_map.shape[0], 
                              0:full_segmentation_map.shape[1], 
                              0:full_segmentation_map.shape[2]] = full_segmentation_map
    return new_full_img_tensor, new_full_segmentation_map, new_imsize


def pad_bygiven_size(full_img_tensor, full_segmentation_map, pad_size):
    new_imsize = [full_img_tensor.shape[1] + pad_size,
                  full_img_tensor.shape[2] + pad_size,
                  full_img_tensor.shape[3] + pad_size]
    
    new_full_img_tensor = torch.zeros([full_img_tensor.shape[0], 
                                       new_imsize[0], new_imsize[1], new_imsize[2]],
                                      dtype = full_img_tensor.dtype)
    
    new_full_segmentation_map = torch.zeros([new_imsize[0], new_imsize[1], new_imsize[2]],
                                            dtype = torch.int16)
            
    new_full_img_tensor[:, 0:full_img_tensor.shape[1], 
                        0:full_img_tensor.shape[2], 
                        0:full_img_tensor.shape[3]] = full_img_tensor
    new_full_segmentation_map[0:full_segmentation_map.shape[0], 
                              0:full_segmentation_map.shape[1], 
                              0:full_segmentation_map.shape[2]] = full_segmentation_map
    return new_full_img_tensor, new_full_segmentation_map, new_imsize