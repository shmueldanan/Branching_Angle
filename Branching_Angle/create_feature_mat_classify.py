'''
Created on 19 Jul 2020

@author: danan
'''

from skimage.color import rgb2hsv
from skimage import exposure
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.transform import pyramid_gaussian, rescale, resize
from sklearn import preprocessing
#from sklearn.svm import LinearSVC

#import matplotlib.pyplot as plt
import numpy as np

def create_feature_mat_classify(im_orig, im_red,im,im_blue, numSegments):
    
    # SLIC Algorithm 
     
    
    segments = slic(im_orig, n_segments = numSegments, sigma = 4,compactness=10)   
    
    
    
    # Histogram Equalization
    im_hist_r = exposure.equalize_hist(im_red, nbins=256, mask=None)
    im_hist_g = exposure.equalize_hist(im, nbins=256, mask=None)
    im_hist_b = exposure.equalize_hist(im_blue, nbins=256, mask=None)
    
    hsv_im = rgb2hsv(im_orig)
    hue_img = hsv_im[:, :, 0]
    saturation_img = hsv_im[:, :, 1]
    
    nbins = 256
    Hist_feature_vec = np.zeros((nbins*5, numSegments))
    
    for i in range(numSegments):  
        
        Hist_feature_vec[0:256,i] = np.histogram(im_hist_r[segments==i], bins=nbins, range=(0,1))[0]
        Hist_feature_vec[256:512,i] = np.histogram(im_hist_g[segments==i], bins=nbins, range=(0,1))[0]
        Hist_feature_vec[512:768,i] = np.histogram(im_hist_b[segments==i], bins=nbins, range=(0,1))[0]
        Hist_feature_vec[768:1024,i] = np.histogram(hue_img[segments==i], bins=nbins, range=(0,1))[0]
        Hist_feature_vec[1024:1280,i] = np.histogram(saturation_img[segments==i], bins=nbins, range=(0,1))[0]
    
    print('dfd')
    
    # CSS features
    
    # Pyramid Gaussian
    
    im_red_pyramid = tuple(pyramid_gaussian(im_red, max_layer=8, downscale=2))
    im_green_pyramid = tuple(pyramid_gaussian(im, max_layer=8, downscale=2))
    im_blue_pyramid = tuple(pyramid_gaussian(im_blue, max_layer=8, downscale=2))
    
    # diff maps
    
    diff_maps = np.zeros((im_red.shape[0],im_red.shape[1],18))
    
    diff_maps[:,:,0] = resize(np.abs(im_red_pyramid[2] - resize(im_red_pyramid[5],im_red_pyramid[2].shape)),im_red.shape)
    diff_maps[:,:,1] = resize(np.abs(im_red_pyramid[2] - resize(im_red_pyramid[6],im_red_pyramid[2].shape)),im_red.shape)
    diff_maps[:,:,2] = resize(np.abs(im_red_pyramid[3] - resize(im_red_pyramid[6],im_red_pyramid[3].shape)),im_red.shape)
    diff_maps[:,:,3] = resize(np.abs(im_red_pyramid[3] - resize(im_red_pyramid[7],im_red_pyramid[3].shape)),im_red.shape)
    diff_maps[:,:,4] = resize(np.abs(im_red_pyramid[4] - resize(im_red_pyramid[7],im_red_pyramid[4].shape)),im_red.shape)
    diff_maps[:,:,5] = resize(np.abs(im_red_pyramid[4] - resize(im_red_pyramid[8],im_red_pyramid[4].shape)),im_red.shape)
    
    diff_maps[:,:,6] = resize(np.abs(im_green_pyramid[2] - resize(im_green_pyramid[5],im_green_pyramid[2].shape)),im.shape)
    diff_maps[:,:,7] = resize(np.abs(im_green_pyramid[2] - resize(im_green_pyramid[6],im_green_pyramid[2].shape)),im.shape)
    diff_maps[:,:,8] = resize(np.abs(im_green_pyramid[3] - resize(im_green_pyramid[6],im_green_pyramid[3].shape)),im.shape)
    diff_maps[:,:,9] = resize(np.abs(im_green_pyramid[3] - resize(im_green_pyramid[7],im_green_pyramid[3].shape)),im.shape)
    diff_maps[:,:,10] = resize(np.abs(im_green_pyramid[4] - resize(im_green_pyramid[7],im_green_pyramid[4].shape)),im.shape)
    diff_maps[:,:,11] = resize(np.abs(im_green_pyramid[4] - resize(im_green_pyramid[8],im_green_pyramid[4].shape)),im.shape)
    
    diff_maps[:,:,12] = resize(np.abs(im_blue_pyramid[2] - resize(im_blue_pyramid[5],im_blue_pyramid[2].shape)),im_blue.shape)
    diff_maps[:,:,13] = resize(np.abs(im_blue_pyramid[2] - resize(im_blue_pyramid[6],im_blue_pyramid[2].shape)),im_blue.shape)
    diff_maps[:,:,14] = resize(np.abs(im_blue_pyramid[3] - resize(im_blue_pyramid[6],im_blue_pyramid[3].shape)),im_blue.shape)
    diff_maps[:,:,15] = resize(np.abs(im_blue_pyramid[3] - resize(im_blue_pyramid[7],im_blue_pyramid[3].shape)),im_blue.shape)
    diff_maps[:,:,16] = resize(np.abs(im_blue_pyramid[4] - resize(im_blue_pyramid[7],im_blue_pyramid[4].shape)),im_blue.shape)
    diff_maps[:,:,17] = resize(np.abs(im_blue_pyramid[4] - resize(im_blue_pyramid[8],im_blue_pyramid[4].shape)),im_blue.shape)
    
    
    CSS_mat= np.zeros((180, numSegments))
    
    for i in range(numSegments):
        index_array = np.argwhere(segments==i)
        for j in range(0,18):
            
            if not index_array.shape[0]==0:
                CSS_mat[j*10,i]= np.mean(diff_maps[:,:,j][segments==i])
                CSS_mat[j*10+1,i]= np.sum(np.power(diff_maps[:,:,j][segments==i] - np.mean(diff_maps[:,:,j][segments==i]),2))/len(diff_maps[:,:,j][segments==i])
                
                # SP Neighbors
                
                
                # Top
                if index_array[np.argmin(index_array[:,0])][0]-1 >= 0:
                    SP_i = segments[index_array[np.argmin(index_array[:,0])][0]- 1,index_array[np.argmin(index_array[:,0])][1]]
                    CSS_mat[j*10+2,i]= np.mean(diff_maps[:,:,j][segments==SP_i])
                    CSS_mat[j*10+3,i]= np.sum(np.power(diff_maps[:,:,j][segments==SP_i] - np.mean(diff_maps[:,:,j][segments==SP_i]),2))/len(diff_maps[:,:,j][segments==SP_i])
                else:
                    CSS_mat[j*10+2,i]= 0
                    CSS_mat[j*10+3,i]= 0
                
                # Bottom
                if index_array[np.argmax(index_array[:,0])][0]+1 < segments.shape[0]:
                    SP_i = segments[index_array[np.argmax(index_array[:,0])][0]+ 1 ,index_array[np.argmax(index_array[:,0])][1]]
                    CSS_mat[j*10+4,i]= np.mean(diff_maps[:,:,j][segments==SP_i])
                    CSS_mat[j*10+5,i]= np.sum(np.power(diff_maps[:,:,j][segments==SP_i] - np.mean(diff_maps[:,:,j][segments==SP_i]),2))/len(diff_maps[:,:,j][segments==SP_i])
                else:
                    CSS_mat[j*10+4,i]= 0
                    CSS_mat[j*10+5,i]= 0
                
                # Left
                if index_array[np.argmin(index_array[:,1])][1]-1 >= 0:
                    SP_i = segments[index_array[np.argmin(index_array[:,1])][0], index_array[np.argmax(index_array[:,1])][1]- 1]
                    CSS_mat[j*10+6,i]= np.mean(diff_maps[:,:,j][segments==SP_i])
                    CSS_mat[j*10+7,i]= np.sum(np.power(diff_maps[:,:,j][segments==SP_i] - np.mean(diff_maps[:,:,j][segments==SP_i]),2))/len(diff_maps[:,:,j][segments==SP_i])
                else:
                    CSS_mat[j*10+6,i]= 0
                    CSS_mat[j*10+7,i]= 0
                
                # Right    
                if index_array[np.argmax(index_array[:,1])][1]+1 < segments.shape[1]:
                    SP_i = segments[index_array[np.argmax(index_array[:,1])][0], index_array[np.argmax(index_array[:,1])][1]+ 1]
                    CSS_mat[j*10+8,i]= np.mean(diff_maps[:,:,j][segments==SP_i])
                    CSS_mat[j*10+9,i]= np.sum(np.power(diff_maps[:,:,j][segments==SP_i] - np.mean(diff_maps[:,:,j][segments==SP_i]),2))/len(diff_maps[:,:,j][segments==SP_i])
                else:
                    CSS_mat[j*10+8,i]= 0
                    CSS_mat[j*10+9,i]= 0
            else:
                CSS_mat[j*10,i]= 0
                CSS_mat[j*10+1,i]= 0
                CSS_mat[j*10+2,i]= 0
                CSS_mat[j*10+3,i]= 0
                CSS_mat[j*10+4,i]= 0
                CSS_mat[j*10+5,i]= 0
                CSS_mat[j*10+6,i]= 0
                CSS_mat[j*10+7,i]= 0
                CSS_mat[j*10+8,i]= 0
                CSS_mat[j*10+9,i]= 0                 
     
    # Concatenate and Normalize matrix
    feature_mat = np.concatenate((Hist_feature_vec,CSS_mat)).T
    feature_mat_norm = preprocessing.normalize(feature_mat, norm = 'l1')   
    
    
    
    return feature_mat_norm, segments