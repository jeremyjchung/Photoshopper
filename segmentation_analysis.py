import cv2
import numpy as np

from skimage.segmentation import slic
from skimage.segmentation import find_boundaries

import greyscale_img
import derivative_analysis

def run_slic(img, n=120, c=40):
    segments = slic(img, n_segments=n, compactness=c)
    return segments

def run_find_boundaries(img):
    boundaries = find_boundaries(img)
    return boundaries

def slic_averager(img, segments):
    h = {}
    matrix = np.zeros(img.shape)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            label = segments[i,j]
            if label in h:
                h[label] = (h[label][0] + img[i,j], h[label][1] + 1)
            else:
                h[label] = (img[i,j], 1)

    avgs = {}
    var = {}

    for key, value in h.items():
        avgs[key] = value[0] / value[1]
        var[key] = 0

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            l = segments[i,j]
            matrix[i,j] = avgs[l]
            var[l] = var[l] + (img[i,j] - avgs[l])**2

    for key, value in var.items():
        var[key] = value / (h[key][1] - 1)

    return matrix, var

def intensity_map(img):
    matrix = np.full((img.shape[0],img.shape[1]), 255)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            v = (img[i,j,0]**2 + img[i,j,1]**2 + img[i,j,2]**2)**0.5
            matrix[i,j] = 255 - v

    return matrix

def detail_level_map(img, segments, var_hash, threshold=75):
    matrix = np.full((img.shape[0],img.shape[1]), 255)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            v = var_hash[segments[i,j]]
            if v >= threshold:
                matrix[i,j] = img[i,j]

    return matrix

def segmentation_analysis():
    img = cv2.imread("./images/birdo.jpg", cv2.IMREAD_UNCHANGED)
    grey_img = greyscale_img.generate(img)

    #segments = run_slic(grey_img)
    #slic_avg_img, var_hash = slic_averager(grey_img, segments)
    #detail_level_img = detail_level_map(grey_img, segments, var_hash)

    segments_color = run_slic(img)
    segments_grey = run_slic(grey_img)
    slic_avg_img_color, _ = slic_averager(grey_img, segments_color)
    slic_avg_img_grey, _ = slic_averager(grey_img, segments_grey)

    #intensity_img = intensity_map(img)

    cv2.imwrite("./outputs/birdo_slic_avg.png", slic_avg_img_grey)
    #cv2.imwrite("./outputs/birdo_detail_level_map.png", detail_level_img)
    cv2.imwrite("./outputs/birdo_slic_avg_color.png", slic_avg_img_color)
    #cv2.imwrite("./outputs/birdo_intensity_map.png", intensity_img)



if __name__ == '__main__':
    segmentation_analysis()
