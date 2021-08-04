import numpy as np
import scipy.ndimage as ndimage
from lib.medloaders.medical_image_process import rescale_data_volume


def random_zoom(img_numpy, min_percentage=0.8, max_percentage=1.1):
    """
    :param img_numpy: 
    :param min_percentage: 
    :param max_percentage: 
    :return: zoom in/out aigmented img
    """
    z = np.random.sample() * (max_percentage - min_percentage) + min_percentage
    zoom_matrix = np.array([[z, 0, 0, 0],
                            [0, z, 0, 0],
                            [0, 0, z, 0],
                            [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(img_numpy, zoom_matrix)


class RandomZoom(object):
    def __init__(self, min_percentage=0.8, max_percentage=1.1):
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        initial_size = np.shape(img_numpy)
        img_numpy = random_zoom(img_numpy, self.min_percentage, self.max_percentage)
        if label.any() != None:
            label = random_zoom(label, self.min_percentage, self.max_percentage)
        img_numpy = rescale_data_volume(img_numpy, initial_size)
        label = rescale_data_volume(label, initial_size)
        return img_numpy, label
