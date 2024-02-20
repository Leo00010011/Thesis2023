import cv2
import numpy as np

def get_img_real_size(img:np.ndarray):
    h = img[:,0,0].argmin()
    w = img[0,:,0].argmin()
    return h, w

def get_cropped_img(img):
    h, w = get_img_real_size(img)
    return img[0:h,0:w,:]

