from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
import numpy as np


def enhance_mask(mask: np.ndarray):
    mask = np.uint(mask > .5)
    mask = binary_fill_holes(mask > 0).astype(float)
    return remove_small_objects(mask > .5, min_size=100, connectivity=2)
