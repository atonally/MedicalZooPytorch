import nibabel as nib
import SimpleITK as sitk
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

"""
concentrate all pre-processing here here
"""


def load_medical_image(path, type=None, reference_image = None, resample=None,
                       viz3d=False, to_canonical=True, rescale=None, normalization='full_volume_mean',
                       clip_intenisty=False, crop_size=(0, 0, 0), crop=(0, 0, 0), fillEmpty = True):
    
    if 'VISCERAL' in path or 'BTCV' in path or 'pancreasCT' in path:
        img_nii = nib.load(path)
        voxel_size = img_nii.header.get_zooms()
        if to_canonical:
             img_nii = nib.as_closest_canonical(img_nii)
               
        img_info = img_nii.get_affine()
        img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))
        if type!='label' and fillEmpty:
            img_np = fill_empty_slices(img_np)
        orig_imsize = img_np.shape
        
        if resample is not None: 
            new_shape = np.round(((np.array(voxel_size) / np.array(resample)).astype(float) * np.asarray(img_np.shape))).astype(int)
        
            if type!='label':
                img_np = rescale_data_volume(img_np, new_shape, interp_order=3)
            else:
                img_np = rescale_data_volume(img_np, new_shape, interp_order=0)
                
    elif reference_image is not None:
        input_reader = sitk.ImageFileReader()
        input_reader.SetFileName(reference_image)
        ref_image = input_reader.Execute()
        orig_imsize = ref_image.GetSize()
        
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetReferenceImage(ref_image)        
        ## other im
        input_reader.SetFileName(path)
        image = input_reader.Execute()
        
        if type!='label' and fillEmpty:
            img_np = sitk.GetArrayFromImage(image)
            img_np = np.moveaxis(img_np, 0, -1)
            img_np = fill_empty_slices(img_np)
            img_np = np.moveaxis(img_np, -1, 0)
            image_new = sitk.GetImageFromArray(img_np)
            image_new.CopyInformation(image)
            image = image_new
        
        if type == 'label':
            resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
            
        if resample is not None: 
            original_spacing = ref_image.GetSpacing()
            original_size = ref_image.GetSize()
            out_spacing = resample
            out_size = [
                int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
            resample_filter.SetOutputSpacing(resample)
            resample_filter.SetSize(np.array(out_size).tolist())
            
        img_info = resample_filter.Execute(image)
        
        img_np = sitk.GetArrayFromImage(img_info)
        img_np = np.moveaxis(img_np, 0, -1)
        img_np = np.fliplr(img_np)
        img_np = np.rot90(img_np, 3)
    else: 
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        img_info = reader.Execute()
        if type!='label' and fillEmpty:
            img_np = sitk.GetArrayFromImage(img_info)
            img_np = np.moveaxis(img_np, 0, -1)
            img_np = fill_empty_slices(img_np)
            img_np = np.moveaxis(img_np, -1, 0)
            image_new = sitk.GetImageFromArray(img_np)
            image_new.CopyInformation(img_info)
            img_info = image_new
            
        orig_imsize = img_info.GetSize()
        
        if resample is not None: 
            original_spacing = img_info.GetSpacing()
            original_size = img_info.GetSize()
            out_spacing = resample
            out_size = [
                int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
            resample_filter = sitk.ResampleImageFilter()
            resample_filter.SetOutputSpacing(resample)
            resample_filter.SetSize(np.array(out_size).tolist())
            resample_filter.SetOutputDirection(img_info.GetDirection())
            resample_filter.SetOutputOrigin(img_info.GetOrigin())
            resample_filter.SetTransform(sitk.Transform())
            resample_filter.SetDefaultPixelValue(img_info.GetPixelIDValue())
            if type=='label':
                resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
            else:
                resample_filter.SetInterpolator(sitk.sitkBSpline)
            img_info_new = resample_filter.Execute(img_info)
        else:
            img_info_new = img_info
        #img_info = img_info_new    
        img_np = sitk.GetArrayFromImage(img_info_new)
        
        img_np = np.moveaxis(img_np, 0, -1)
        img_np =  np.fliplr(img_np)
        img_np = np.rot90(img_np, 3)
        
    if type=='label':
        img_np = img_np.astype(np.int16)
    else:
        img_np = img_np.astype(np.float32)
        
        
    
    if type!='label' and fillEmpty:
        img_np = fill_empty_slices(img_np)
    
    if type!='label':
        img_np[img_np<-1000] = -1000 # I do this because sometimes there are values of -3024 in CT images, and that can confuse the normalization and network
    
    if viz3d:
        return img_np, img_info, orig_imsize
        
    # 3. intensity normalization
    img_tensor = torch.from_numpy(img_np.copy())
    
    MEAN, STD, MAX, MIN = 0., 1., 1., 0.
    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()
    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
    if crop is not None:
        img_tensor = crop_img(img_tensor, crop_size, crop)
        
    return img_tensor, img_info, orig_imsize


def medical_image_transform(img_tensor, type=None,
                            normalization="full_volume_mean",
                            norm_values=(0., 1., 1., 0.)):
    MEAN, STD, MAX, MIN = norm_values
    # Numpy-based transformations/augmentations here

    if type != 'label':
        MEAN, STD = img_tensor.mean(), img_tensor.std()
        MAX, MIN = img_tensor.max(), img_tensor.min()

    if type != "label":
        img_tensor = normalize_intensity(img_tensor, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))

    return img_tensor


def crop_img(img_tensor, crop_size, crop):
    if crop_size[0] == 0:
        return img_tensor
    slices_crop, w_crop, h_crop = crop
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_tensor.dim()
    assert inp_img_dim >= 3
    if img_tensor.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img_tensor.shape
    elif img_tensor.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img_tensor.shape
        img_tensor = img_tensor[0, ...]
    if full_dim1 == dim1 and full_dim2 == dim2 and full_dim3 == dim3:
        img_tensor = img_tensor
    elif full_dim1 == dim1 and full_dim2 == dim2:
        img_tensor = img_tensor[:, :,
                     h_crop:h_crop + dim3]
    elif full_dim1 == dim1 and full_dim3 == dim3:
        img_tensor = img_tensor[:, w_crop:w_crop+dim2, :]
    elif full_dim2 == dim2 and full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :, :]
    elif full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                     h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_tensor.unsqueeze(0)

    return img_tensor


def load_affine_matrix(path):
    """
    Reads an path to nifti file and returns the affine matrix as numpy array 4x4
    """
    img = nib.load(path[0])
    return img.affine


def load_2d_image(img_path, resize_dim=0, type='RGB'):
    image = Image.open(img_path)
    if type == 'RGB':
        image = image.convert(type)
    if resize_dim != 0:
        image = image.resize(resize_dim)
    pix = np.array(image)
    return pix


def rescale_data_volume(img_numpy, out_dim, interp_order=0):
    """
    Resize the 3d numpy array to the dim size
    :param out_dim is the new 3d tuple
    """
    if img_numpy.ndim == 2:
        depth, height = img_numpy.shape
        scale = [out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height]
    else:
        depth, height, width = img_numpy.shape
        scale = [out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height, out_dim[2] * 1.0 / width]
    return ndimage.interpolation.zoom(img_numpy, scale, order = interp_order)


def transform_coordinate_space(img2, aff2, aff1, out_shape):
    """
    Accepts nifty objects
    Transfers coordinate space from modality_2 to modality_1
    """
    #aff_t1 = modality_1.affine
    #aff_t2 = modality_2.affine
    inv_af_2 = np.linalg.inv(aff2)

    #out_shape = modality_1.get_fdata().shape

    # desired transformation
    T = inv_af_2.dot(aff1)
    transformed = ndimage.affine_transform(img2, T, output_shape=out_shape, order=0)

    return transformed


def normalize_intensity(img_tensor, normalization="full_volume_mean", norm_values=(0, 1, 1, 0)):
    """
    Accepts an image tensor and normalizes it
    :param normalization: choices = "max", "mean" , type=str
    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        max_val, _ = torch.max(img_tensor)
        img_tensor = img_tensor / max_val
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]

    elif normalization == 'max_min':
        img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor

def fill_empty_slices(img_numpy):
    img_numpy[img_numpy<-1000] = -1000
    max_per_slice = np.max(np.max(img_numpy, axis=0), axis=0)
    #empty_slices = [xx for xx in range(1, len(max_per_slice)-1) if abs(max_per_slice[xx+1]-max_per_slice[xx])>500]
    empty_slices = [xx for xx in range(1, len(max_per_slice)-1) if max_per_slice[xx] <= -1000 or max_per_slice[xx]>=5000] #and max_per_slice[xx+1] != -1000 and max_per_slice[xx-1] != -1000]
    
    double_empty_slices = [xx for xx in range(2, len(max_per_slice)-2) if max_per_slice[xx] == -1000 and xx not in empty_slices]
    for sl in empty_slices: 
        if sl>0 and  sl<len(max_per_slice):
            img_numpy[:,:,sl] = (img_numpy[:,:,sl-1] + img_numpy[:,:,sl+1]) / 2
        
    for xx in range(len(double_empty_slices)//2) :
        if ((double_empty_slices[xx]-1)>0) and ((double_empty_slices[xx]+1)<len(max_per_slice)):
            sl = double_empty_slices[xx*2]
            img_numpy[:,:,sl] = (0.75*img_numpy[:,:,sl-1] + 0.25*img_numpy[:,:,sl+2]) 
            img_numpy[:,:,sl+1] = (0.25*img_numpy[:,:,sl-1] + 0.75*img_numpy[:,:,sl+2]) 
       

    return img_numpy

def clip_range(img_numpy):
    """
    Cut off outliers that are related to detected black in the image (the air area)
    """
    # Todo median value!
    #zero_value = (img_numpy[0, 0, 0] + img_numpy[-1, 0, 0] + img_numpy[0, -1, 0] + \
    #              img_numpy[0, 0, -1] + img_numpy[-1, -1, -1] + img_numpy[-1, -1, 0] \
    #              + img_numpy[0, -1, -1] + img_numpy[-1, 0, -1]) / 8.0
    non_zeros_idx = np.where(img_numpy > -1024)
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    y = img_numpy[min_z:max_z, min_h:max_h, min_w:max_w]
    return y


def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy
