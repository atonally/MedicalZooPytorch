import numpy as np
import scipy.ndimage as ndimage
from lib.medloaders.medical_image_process import rescale_data_volume

def random_rotate3D(img_numpy, label, min_angle, max_angle):
    """
    Returns a random rotated array in the same shape
    :param img_numpy: 3D numpy array
    :param min_angle: in degrees
    :param max_angle: in degrees
    :return: 3D rotated img
    """
    assert img_numpy.ndim == 3, "provide a 3d numpy array"
    assert min_angle < max_angle, "min should be less than max val"
    assert min_angle > -360 or max_angle < 360
    all_axes = [(1, 0), (1, 2), (0, 2)]
    angle = np.random.randint(low=min_angle, high=max_angle + 1)
    axes_random_id = np.random.randint(low=0, high=len(all_axes))
    axes = all_axes[axes_random_id]
    if label.any() != None:
        label = ndimage.rotate(label, angle, axes=axes, reshape = False, order=0)
    numpy_rotated = ndimage.rotate(img_numpy, angle, axes=axes, reshape = False)
    return numpy_rotated, label


class RandomRotation(object):
    def __init__(self, min_angle=-20, max_angle=20):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img_numpy, label=None):
        """
        Args:
            img_numpy (numpy): Image to be rotated.
            label (numpy): Label segmentation map to be rotated

        Returns:
            img_numpy (numpy): rotated img.
            label (numpy): rotated Label segmentation map.
        """
        initial_size = np.shape(img_numpy)
        img_numpy, label = random_rotate3D(img_numpy, label, self.min_angle, self.max_angle)
        
        img_numpy = rescale_data_volume(img_numpy, initial_size).astype(np.float32)
        label = rescale_data_volume(label, initial_size)
        return img_numpy, label
