import math
from medzoopytorch.medloaders import medical_image_process as img_loader
from medzoopytorch.utils import split_input

import nibabel as nib
import torch
import numpy as np

from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage as ndimage
import SimpleITK as sitk
import torch.nn as nn

def roundup(x, base=32):
    return int(math.ceil(x / base)) * base

def grid_sampler_sub_volume_reshape(tensor, dim):
    return tensor.view(-1, dim[0], dim[1], dim[2])


def save_3d_vol(predictions, img_info, save_path, integer=True, fileExtension='.nii.gz'):
    if isinstance(img_info, np.ndarray):
        pred_nifti_img = nib.Nifti1Image(predictions, img_info)
        pred_nifti_img.header["qform_code"] = 1
        pred_nifti_img.header['sform_code'] = 0
        nib.save(pred_nifti_img, save_path + fileExtension)
        print('3D vol saved')
    else:
        predictions = np.rot90(predictions, 1)
        predictions = np.fliplr(predictions)
        predictions = np.moveaxis(predictions, -1, 0)
        
        Mask_img = sitk.GetImageFromArray(predictions)
        Mask_img.CopyInformation(img_info)
        if integer:
            Mask_img = sitk.Cast(Mask_img,sitk.sitkUInt8)
        ifw = sitk.ImageFileWriter()
        ifw.SetFileName(save_path + fileExtension)
        ifw.SetUseCompression(True)
        ifw.Execute(Mask_img)
    
def get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def compute_steps_for_sliding_window(patch_size, image_size, step_size):
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

def largest_connect_component(segmentation, organs):
    new_segmentation = np.zeros(segmentation.shape)
    for organ in range(len(organs)):
        if organs[organ] in [1, 2, 6, 7, 8, 9, 16, 17, 18]:
            pred_org = segmentation==(organ+1)
            if np.any(pred_org>0):
                label_im, nb_labels = ndimage.label(pred_org)
                sizes = ndimage.sum(pred_org, label_im, range(nb_labels + 1))
                new_segmentation += (organ+1)*(label_im == sizes.argmax())
        elif organs[organ] in [3, 4, 5]: 
            pred_org = segmentation==(organ+1)
            if np.any(pred_org>0):
                label_im, nb_labels = ndimage.label(pred_org)
                sizes = ndimage.sum(pred_org, label_im, range(nb_labels + 1))
                sizes_order = np.argsort(sizes)[::-1]
                new_segmentation += (organ+1)*(label_im == sizes_order[0])
                if len(sizes_order)>2:
                    new_segmentation += (organ+1)*(label_im == sizes_order[1])
        else:
            new_segmentation += (organ+1)*(segmentation==(organ+1))
    return new_segmentation

def largest_bone_segment(segmentation, organs):
    new_segmentation = np.zeros(segmentation.shape)
    for organ in range(len(organs)):
        if organs[organ] in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 26, 27]:
            pred_org = segmentation==(organ+1)
            if np.any(pred_org>0):
                label_im, nb_labels = ndimage.label(pred_org)
                sizes = ndimage.sum(pred_org, label_im, range(nb_labels + 1))
                new_segmentation += (organ+1)*(label_im == sizes.argmax())
        elif organs[organ] in [17, 19, 20, 21, 22, 23]: 
            pred_org = segmentation==(organ+1)
            if np.any(pred_org>0):
                label_im, nb_labels = ndimage.label(pred_org)
                sizes = ndimage.sum(pred_org, label_im, range(nb_labels + 1))
                sizes_order = np.argsort(sizes)[::-1]
                new_segmentation += (organ+1)*(label_im == sizes_order[0])
                if len(sizes_order)>2:
                    new_segmentation += (organ+1)*(label_im == sizes_order[1])
        else:
            new_segmentation += (organ+1)*(segmentation==(organ+1))
    return new_segmentation

def remap_seg_map(segmentation,organs):
    new_segmentation = np.zeros(segmentation.shape)
    for organ in range(len(organs)):
        new_segmentation[segmentation==organ+1] = organs[organ]
    return new_segmentation
    
def overlap_padding(args, full_volume, img_info, orig_imsize, idx, subID,  model, criterion, kernel_dim, saveFile = True):
    
    with torch.no_grad():
        
        data_nopad = full_volume[:,...].detach()
        full_volume_shape = full_volume.shape
        
        """
        data_nopad = data_nopad.cpu().numpy()
        data_nopad_new = data_nopad[:,::-1,:,:]
        data_nopad = torch.from_numpy(data_nopad_new.copy())
        """
        
        #target_nopad = full_volume[-1,...].unsqueeze(0).detach()
        if data_nopad.shape[3] < kernel_dim[2]:
        # find pad values
            data = torch.zeros([data_nopad.shape[0], data_nopad.shape[1], data_nopad.shape[2], kernel_dim[2]])        
            data[:, :, :, 0:data_nopad.shape[3]] = data_nopad
            data = data.cuda()
        else:
            data = data_nopad.cuda()
        if args.model=='DEEPMEDIC':
            data = torch.zeros([data_nopad.shape[0], data_nopad.shape[1]+128, data_nopad.shape[2]+128, data_nopad.shape[3]+128])        
            data[:, 64:data_nopad.shape[1]+64, 64:data_nopad.shape[2]+64, 64:data_nopad.shape[3]+64] = data_nopad
            data = data.cuda()
        #data = torch.flip(data, [2])  
        
        modalities, D, H, W = full_volume.shape
        patch_size = kernel_dim
        step_size = args.overlap_stepsize
        
        # compute the steps for sliding window
        steps = compute_steps_for_sliding_window(patch_size, data.shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        print("data shape:", data.shape)
        print("patch size:", patch_size)
        print("steps (x, y, and z):", steps)
        print("number of tiles:", num_tiles)

    
        if args.model == 'DEEPMEDIC':
            patch_size_output = [45, 45, 45]
        else:
            patch_size_output = patch_size
        
        gaussian_importance_map = get_gaussian(patch_size_output, sigma_scale=1. / 8)

        aggregated_results = np.zeros([args.classes, data.shape[1], data.shape[2], data.shape[3]], dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros(data.shape[1:], dtype=np.float32)
     
        batch_size = args.batchSz_test
        batch = np.zeros([batch_size, modalities, patch_size[0], patch_size[1], patch_size[2]], dtype=np.float32)
        batch_locs = np.zeros([batch_size, 6], dtype = int)
        batch = torch.from_numpy(batch).cuda()
                
        batches_complete = 0
        batches_total = np.ceil(num_tiles/batch_size)
        bn = 0
        del full_volume
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]
                    
                    patch = data[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z]
                    batch_locs[bn, :] = [lb_x, ub_x, lb_y, ub_y, lb_z, ub_z]
                    batch[bn, :, :, :, :] = patch
                    bn += 1
                    if bn == batch_size:
                        torch.cuda.empty_cache()
                        bn=0
                        if args.model == 'DEEPMEDIC':
                            batch_in, target = split_input(batch, target=None)
                            predicted_batch = model(batch_in)
                        else:
                            predicted_batch = model(batch)

                        batches_complete += 1
                        predicted_batch = predicted_batch.cpu().numpy()
                        torch.cuda.empty_cache()

                        for pp in range(batch_size):
                            predicted_patch = predicted_batch[pp,:,:,:,:]
                            predicted_patch *= gaussian_importance_map          
                            if np.any(batch_locs[pp,:]>0):
                                lb_x, ub_x, lb_y, ub_y, lb_z, ub_z = batch_locs[pp,  :]
                                if args.model == 'DEEPMEDIC':
                                    lb_x1 = lb_x + int(patch_size[0]/2 - 45/2)+1
                                    ub_x1 = ub_x - int( patch_size[0]/2 - 45/2)
                                    lb_y1 = lb_y + int(patch_size[1]/2 - 45/2)+1
                                    ub_y1 = ub_y - int( patch_size[1]/2 - 45/2)
                                    lb_z1 = lb_z + int(patch_size[2]/2 - 45/2)+1
                                    ub_z1 = ub_z - int( patch_size[2]/2 - 45/2)
                                else:
                                    lb_x1, ub_x1, lb_y1, ub_y1, lb_z1, ub_z1 = batch_locs[pp,  :]
    
                                aggregated_results[:, lb_x1:ub_x1, lb_y1:ub_y1, lb_z1:ub_z1] += predicted_patch.squeeze()
                                aggregated_nb_of_predictions[lb_x1:ub_x1, lb_y1:ub_y1, lb_z1:ub_z1] += gaussian_importance_map
                    
                    if batches_complete==batches_total-1 and bn == num_tiles - batches_complete*batch_size:
                        if args.model == 'DEEPMEDIC':
                            batch_in, target = split_input(batch, target=None)
                            predicted_batch = model(batch_in)
                        else:
                            predicted_batch = model(batch)
                        
                        predicted_batch = predicted_batch.cpu().numpy()
                        torch.cuda.empty_cache()

                        for pp in range(batch_size):
                            predicted_patch = predicted_batch[pp,:,:,:,:]
                            predicted_patch *= gaussian_importance_map          
                            if np.any(batch_locs[pp,:]>0):
                                lb_x, ub_x, lb_y, ub_y, lb_z, ub_z = batch_locs[pp,  :]
                                if args.model == 'DEEPMEDIC':
                                    lb_x1 = lb_x + int(patch_size[0]/2 - 45/2)+1
                                    ub_x1 = ub_x - int( patch_size[0]/2 - 45/2)
                                    lb_y1 = lb_y + int(patch_size[1]/2 - 45/2)+1
                                    ub_y1 = ub_y - int( patch_size[1]/2 - 45/2)
                                    lb_z1 = lb_z + int(patch_size[2]/2 - 45/2)+1
                                    ub_z1 = ub_z - int( patch_size[2]/2 - 45/2)
                                else:
                                    lb_x1, ub_x1, lb_y1, ub_y1, lb_z1, ub_z1 = batch_locs[pp,  :]
                                aggregated_results[:, lb_x1:ub_x1, lb_y1:ub_y1, lb_z1:ub_z1] += predicted_patch.squeeze()
                                aggregated_nb_of_predictions[lb_x1:ub_x1, lb_y1:ub_y1, lb_z1:ub_z1] += gaussian_importance_map
        del batch, data        
        torch.cuda.empty_cache()
        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        class_probabilities = aggregated_results / aggregated_nb_of_predictions
        class_probabilities = class_probabilities.astype(np.float32)
        
        if args.model=='DEEPMEDIC':
            class_probabilities = class_probabilities[:, 64:, 64:, 64:]
            class_probabilities = class_probabilities[:, :(class_probabilities.shape[1]-64), :(class_probabilities.shape[2]-64), :(class_probabilities.shape[3]-64)]
        class_probabilities = class_probabilities[:,0:full_volume_shape[1], 0:full_volume_shape[2], 0:full_volume_shape[3]]
        
        
        new_probs = np.zeros([class_probabilities.shape[0], orig_imsize[0], orig_imsize[1], orig_imsize[2]])
        for i in range(new_probs.shape[0]):
            new_probs[i,:,:,:] = img_loader.rescale_data_volume(class_probabilities[i,:,:,:], orig_imsize,interp_order=1)
        
        if args.saveProbabilityMaps:
            # Normalize probability maps to 1
            m = nn.Softmax(dim=0)
            inputs = torch.from_numpy(new_probs)
            probs_norm = m(inputs).numpy()
            # right now this is hard coded for class 1 because that's what I need - can be changed later
            save_path = args.save + '/probabilitymaps/' + subID + '_class1'
            save_3d_vol(probs_norm[1,:], img_info, save_path, integer = False)
        
        output = new_probs.argmax(0) 
        #output = np.flip(output, axis=1)
        if args.largestComponent and args.training_type == 'CT_organSeg':
            output = largest_connect_component(output, args.organs)
        if args.largestComponent and args.training_type == 'CT_boneSeg':
            output = largest_bone_segment(output, args.organs)
        output = remap_seg_map(output, args.organs)
        output = output.astype(np.int16)
        
        if saveFile:
            save_path = args.save + '/' + subID + '_predicted'
            save_3d_vol(output, img_info, save_path)
        return output

     
