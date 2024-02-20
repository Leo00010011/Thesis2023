import numpy as np
import cv2


# Focus
KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def focus(img):
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=np.std(img))
    return cv2.filter2D(img, -1, KERNEL)

# Contrast

def CLAHE(rgb_img):
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def image_agcwd(img, a=0.25, truncated_cdf=False):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()

    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0]**a)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0])**a))
    prob_normalized_wd = pn_temp / pn_temp.sum()  # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

    if truncated_cdf:
        inverse_cdf = np.maximum(0.5, 1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd

    img_new = img.copy()
    for i in unique_intensity:
        img_new[img == i] = np.round(255 * (i / 255)**inverse_cdf[i])

    return img_new


def process_bright(rgb_img):
    img_negative = 255 - rgb_img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reversed = 255 - agcwd
    return reversed


def process_dimmed(rgb_img):
    agcwd = image_agcwd(rgb_img, a=0.75, truncated_cdf=True)
    return agcwd


def contrast_pipeline(rgb_img):
    rgb_img = CLAHE(rgb_img)
    YCrCb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    Y = YCrCb[:, :, 0]
    # Determine whether image is bright or dimmed
    threshold = 0.3
    exp_in = 112  # Expected global average intensity
    M, N = rgb_img.shape[:2]
    mean_in = np.sum(Y/(M*N))
    t = (mean_in - exp_in) / exp_in
    if t < -threshold:  # Dimmed Image
        result = process_dimmed(Y)
        YCrCb[:, :, 0] = result
        return cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)
    elif t > threshold:
        result = process_bright(Y)
        YCrCb[:, :, 0] = result
        return cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)
    return rgb_img

def enhance_image_quality(rgb_img):
    rgb_img = focus(rgb_img)
    rgb_img = contrast_pipeline(rgb_img)
    return rgb_img 