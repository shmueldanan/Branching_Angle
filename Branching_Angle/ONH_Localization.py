'''
Created on 25 Oct 2020

@author: danan
'''
import cv2
import numpy as np
from skimage.measure import  regionprops
from skimage.segmentation import flood, flood_fill


def ONH_Localization(im_orig):
    
    im = im_orig[:,:,1]
    im_red = im_orig[:,:,0]
    im_blue = im_orig[:,:,2]

    
    im_copy= im.copy()
    Threshold = 20
    if (im > Threshold).sum()/im.size < 0.01:
        while  (im>Threshold).sum()/im.size < 0.01:
            Threshold = Threshold-1
    else:
        while  (im>Threshold).sum()/im.size > 0.01:
            Threshold = Threshold+1
    
    Binary_thresh = im > Threshold
    
    ret, labels = cv2.connectedComponents(Binary_thresh.astype(np.uint8))
    regions = regionprops(labels)
    
    region_area_threshold = (im.shape[0]*im.shape[1])/1900
    
    for region in regions:
        #print(region.label)
        #print(region.area)
        if region.area<region_area_threshold :
            labels[labels==region.label] = 0
        elif region.major_axis_length/region.minor_axis_length > 5:    
            im_copy = flood_fill(im_copy, (int(region.centroid[0]), int(region.centroid[1])), 0, tolerance=40) 
            labels[labels==region.label] = 0
    
    Threshold = 20
    if (im_copy > Threshold).sum()/im_copy.size < 0.01:
        while  (im_copy>Threshold).sum()/im_copy.size < 0.01:
            Threshold = Threshold-1
    else:
        while  (im_copy>Threshold).sum()/im_copy.size > 0.01:
            Threshold = Threshold+1
    
    Binary_thresh = im_copy > Threshold 
    
    ret, labels = cv2.connectedComponents(Binary_thresh.astype(np.uint8))
    regions = regionprops(labels)
    
    region_value_max = 0
    region_area_max_label = 0
    for region in regions:
        
        if region.area<region_area_threshold :
            labels[labels==region.label] = 0 
        
        elif np.max(im_copy[labels==region.label])>region_value_max:
            region_value_max = np.max(im_copy[labels==region.label])
            region_area_max_label = region.label
    
    max_ind = np.zeros((2,1))    
    max_ind_0 = int(regions[region_area_max_label-1].centroid[0]) 
    max_ind_1 = int(regions[region_area_max_label-1].centroid[1])
    
    ONH_box_size = round((im.shape[0]+im.shape[1])/16)
    
    im_orig = im_orig[max_ind_0-ONH_box_size:max_ind_0+ONH_box_size, max_ind_1-ONH_box_size:max_ind_1+ONH_box_size,:]
    #im = im[max_ind_0-200:max_ind_0+200, max_ind_1-200:max_ind_1+200]
    #im_red = im_red[max_ind_0-200:max_ind_0+200, max_ind_1-200:max_ind_1+200]
    #im_blue = im_blue[max_ind_0-200:max_ind_0+200, max_ind_1-200:max_ind_1+200]
    
    return im_orig, max_ind_0, max_ind_1, ONH_box_size