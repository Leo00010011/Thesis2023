# import cv2
 
# # Read the original image
# img = cv2.imread('china.jpg') 
# # Display original image
# # cv2.imshow('Original', img)
# # cv2.waitKey(0)
 
# # Convert to graycsale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Blur the image for better edge detection
 
# # Sobel Edge Detection
# sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# # Display Sobel Edge Detection Images
# # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# # cv2.waitKey(0)
# print(sobelxy.max())
# # Canny Edge Detection
# # edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200) # Canny Edge Detection
# # # Display Canny Edge Detection Image
# # cv2.imshow('Canny Edge Detection', edges)
# # cv2.waitKey(0)
 
# # cv2.destroyAllWindows()

from utils.bag_utils import BagReview
from utils.IQA import LAP_MOD
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

b = BagReview('./dec.bag')
def entropy_DMQA(dmap:np.ndarray):
    """
    Depth Map Quality Assesment based on statistics
    """
    # Normalizando el mapa de profundidad
    min_val = np.percentile(dmap,2)
    max_val = np.percentile(dmap,98)
    n_dmap = (dmap - min_val)/(max_val - min_val)
    n_dmap[n_dmap > 1] = 1
    n_dmap[n_dmap < 0] = 0
    # Calculando media y coefficence variance
    md = n_dmap.mean()
    cv = md/n_dmap.std()
    # Calculando entropía
    mask = np.ones(dmap.shape, dtype=np.uint8)*255
    hist = cv2.calcHist([dmap],[0],mask,[256],[min_val,max_val])
    entropy = 0
    total = dmap.shape[0]*dmap.shape[1]
    for index in range(hist.shape[0]):
        p_index = hist[index][0]/total
        if p_index == 0.0:
            continue
        log = math.log2(p_index)
        entropy += (-1)*log*p_index
    
    return math.exp((1 - md)*cv*entropy) - 1
#Falta normalizarlo

def sobel_edge_detector(img:np.ndarray,scale = 4):
    # Sobel edge detector
    min_val = np.percentile(img,2)
    max_val = np.percentile(img,98)    
    n_dmap = (img - min_val)/(max_val - min_val)
    n_dmap[n_dmap > 1] = 1
    n_dmap[n_dmap < 0] = 0
    sobelxy = cv2.Sobel(src=n_dmap, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    cutoff = abs(scale*sobelxy.mean())
    thresh = math.sqrt(cutoff)
    sobelxy[sobelxy > thresh] = 1
    sobelxy[sobelxy < thresh] = 0
    return sobelxy

def canny_edge_detector(img:np.ndarray,upscale = 48, downscale = 60):
    # Canny edge detector
    min_val = np.percentile(img,2)
    max_val = np.percentile(img,98)    
    n_dmap = (img - min_val)/(max_val - min_val)*255
    n_dmap[n_dmap > 255] = 255
    n_dmap[n_dmap < 0] = 0
    n_dmap = n_dmap.astype(np.uint8)
    cutoff = abs(upscale*n_dmap.mean())
    thresh = math.sqrt(cutoff)
    cutoff = abs(downscale*n_dmap.mean())
    thresh2 = math.sqrt(cutoff)
    sobelxy = cv2.Canny(n_dmap,threshold1 = np.uint8(thresh), threshold2= np.uint8(thresh2))
    sobelxy = sobelxy/255
    return sobelxy


def both_over_depth(depth_edges:np.ndarray,color_edges:np.ndarray):
    depth_edge_count = depth_edges.sum()
    inter_edges_count = (color_edges*depth_edges).sum()
    return inter_edges_count/depth_edge_count


def color_over_only_depth_align(depth_edges:np.ndarray,color_edges:np.ndarray):
    # Align function
    color_edges_count = color_edges.sum()
    inter_edges_count = (color_edges*depth_edges).sum()
    only_depth_edges_count = depth_edges.sum() - inter_edges_count
    return only_depth_edges_count/color_edges_count

def color_over_only_depth_REGRESION_align(depth_edges:np.ndarray,color_edges:np.ndarray):
    # Align function with a regressor
    x = color_over_only_depth_align(depth_edges,color_edges)
    a = 0.85
    b = 1.544
    c = 1
    return a/(x**2 + x*b + c)

# Ellos asumen que los bordes de la imagen a color son los reales y los bordes de la de profundidad son los recuperados
# Probar con F1 score(Align function)
# Probar con Dice Coeficient(Align function)



def create_BA_DMQA(edge_func,align_func):
    """
    Depth Map Quality Assesment based in Boundary Alignment
    """
    def BA_DMQA(dmap:np.ndarray,image):
        dmap_edges = edge_func(dmap)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        color_edges = edge_func(gray_image)
        return align_func(dmap_edges,color_edges)
    return BA_DMQA

