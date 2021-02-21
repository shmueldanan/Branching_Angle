'''
Created on 25 Oct 2020

@author: danan
'''
import create_feature_mat_classify as FMC
import cv2 
import numpy as np
from skimage.measure import  regionprops
from skimage.morphology import disk, closing
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter



def ONH_Segmentation(im_orig, model, max_ind_0, max_ind_1,ONH_box_size):

    numSegments = 200

    im = im_orig[:,:,1]
    im_red = im_orig[:,:,0]
    im_blue = im_orig[:,:,2]
    
    feature_mat_norm_test, segments = FMC.create_feature_mat_classify(im_orig, im_red, im, im_blue, numSegments)
    decision_results = model.decision_function(feature_mat_norm_test)
    
    segments2=segments.copy().astype('float')
    for i in range(numSegments):
        segments2[segments==i] = decision_results[i]
    segment_result1 = cv2.blur(segments2,(9,9))
    
    bin_result = segment_result1>0
    
    ret, labels = cv2.connectedComponents(bin_result.astype(np.uint8))
    regions = regionprops(labels)
    
    max_region_area = 0
    for region in regions:
        if region.area>max_region_area :
            max_region_label = region.label
            max_region_area=region.area
    
          
    labels = labels==max_region_label
    
    selem = disk(27)
    label_close = closing(labels+0, selem=selem)
    edges = canny(label_close, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    
    result = hough_ellipse(edges, accuracy=40, threshold=40, min_size=40, max_size=60)      
    result.sort(order='accumulator')
    
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    im_orig[cy, cx] = (0, 0, 255)
    
    cx_real_axes = cx + max_ind_1-ONH_box_size
    cy_real_axes = cy + max_ind_0-ONH_box_size
    
    return cx_real_axes, cy_real_axes