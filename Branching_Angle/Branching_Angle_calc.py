'''
Created on 14 Feb 2021

@author: danan
'''
import numpy as np
import ONH_Localization as ONHLOC
import ONH_Segmentation as ONHSEG
import Create_angle_mask as cmask
import strel_func as strel
import math
from collections import namedtuple


from joblib import dump, load
from skimage import io
import matplotlib.pyplot as plt 
from skimage.measure import  regionprops
from skimage.morphology import skeletonize,thin , medial_axis
from skimage.morphology import erosion,dilation,disk
import cv2

# Load Model

model = load('ONH_model.joblib')

# Read Image

im_orig = io.imread('15_test.tif')
im_orig_vas = io.imread('15_manual1.gif')

im_orig_g = im_orig[:,:,1]

# ONH Algorithm
im_orig_loc, max_ind_0, max_ind_1, ONH_box_size = ONHLOC.ONH_Localization(im_orig)

cx_real_axes, cy_real_axes = ONHSEG.ONH_Segmentation(im_orig_loc, model, max_ind_0, max_ind_1,ONH_box_size)

im_orig[cy_real_axes, cx_real_axes] = (0, 0, 255)

ONH_Radius = ((cy_real_axes.max()-cy_real_axes.min())+(cx_real_axes.max()-cx_real_axes.min()))//4
ONH_Center = [cx_real_axes.min() + (cx_real_axes.max()-cx_real_axes.min())//2 , cy_real_axes.min() + (cy_real_axes.max()-cy_real_axes.min())//2]

Mask = cmask.create_circular_mask(im_orig.shape[0], im_orig.shape[1], center = ONH_Center, radius= ONH_Radius*4.5, radius_small = ONH_Radius*2)

masked_img = np.multiply(im_orig[:,:,1], Mask)
masked_img_vas = np.multiply(im_orig_vas, Mask)

ret, labels = cv2.connectedComponents(masked_img_vas)
regions = regionprops(labels)

for region in regions:

    blob_img = labels==region.label
    blob_skeleton = skeletonize(blob_img)
    
    blob_img_stack = np.uint8(np.dstack((blob_img,blob_img,blob_img)))*255
    blob_skeleton_stack = np.uint8(np.dstack((blob_skeleton,blob_skeleton,blob_skeleton)))*255
    #dst = cv2.cornerHarris(np.float32(blob_skeleton),2,3,0.1)
    dst = cv2.cornerHarris(np.float32(blob_skeleton),2,3,0.1)
    
    blob_skeleton_bifurc_points = np.zeros(blob_skeleton.shape)
    
    skel_pixels = np.where(blob_skeleton)
    bifurcation_labeling = np.zeros(len(skel_pixels[0]))
    
    for k in range(len(skel_pixels[0])):
        if np.sum(blob_skeleton[skel_pixels[0][k]-1:skel_pixels[0][k]+2,skel_pixels[1][k]-1:skel_pixels[1][k]+2])==2:
            bifurcation_labeling[k] = 1
            blob_skeleton_bifurc_points[skel_pixels[0][k],skel_pixels[1][k]] =1 
            blob_img_stack[skel_pixels[0][k],skel_pixels[1][k]] = [255,0,0]
            blob_skeleton_stack[skel_pixels[0][k],skel_pixels[1][k]] = [255,0,0]
            
        elif np.sum(blob_skeleton[skel_pixels[0][k]-1:skel_pixels[0][k]+2,skel_pixels[1][k]-1:skel_pixels[1][k]+2])==4:
            bifurcation_labeling[k] = 2
            blob_skeleton_bifurc_points[skel_pixels[0][k],skel_pixels[1][k]] = 2 
            blob_img_stack[skel_pixels[0][k],skel_pixels[1][k]] = [0,255,0]
            blob_skeleton_stack[skel_pixels[0][k],skel_pixels[1][k]] = [0,255,0]
        elif np.sum(blob_skeleton[skel_pixels[0][k]-1:skel_pixels[0][k]+2,skel_pixels[1][k]-1:skel_pixels[1][k]+2])>4:
            bifurcation_labeling[k] = 3
            blob_skeleton_bifurc_points[skel_pixels[0][k],skel_pixels[1][k]] = 3 
            blob_img_stack[skel_pixels[0][k],skel_pixels[1][k]] = [0,0,255]
            blob_skeleton_stack[skel_pixels[0][k],skel_pixels[1][k]] = [0,0,255]
    
    plt.figure(1)
    plt.imshow(blob_skeleton_stack, cmap='gray')
    
    plt.figure(2)
    plt.imshow(blob_img_stack, cmap='gray')
      
    plt.show()
    
    
    # Disconnected skeleton
    blob_region_skeleton = np.bitwise_xor(np.bitwise_and(blob_skeleton,blob_skeleton_bifurc_points>0),blob_skeleton)
    
    
    ret_skel, labels_skel = cv2.connectedComponents(np.uint8(blob_region_skeleton))
    regions_skel = regionprops(labels_skel)
    
    max_vess = 0
    
    print(region.label)
    
    if regions_skel.__len__()>1 and region.label ==2:
    
        Branch_Struct = namedtuple("Branch_Struct", "branch_points_rows branch_points_cols thickness sgl_blob sgl_blob_skel")
        
        for region_skel in regions_skel:
            
            
            sgl_blob_skel = labels_skel == region_skel.label
            selem = disk(4)
            sgl_blob_skel_dilate = dilation(sgl_blob_skel,selem)
            sgl_blob = np.bitwise_and(sgl_blob_skel_dilate,blob_img)
            
            thickness = np.sum(sgl_blob)/np.sum(sgl_blob_skel)
            print(thickness)
            
            #region_skel.orientation()
                                
            branch_points_rows=np.where(sgl_blob_skel)[0]
            branch_points_cols=np.where(sgl_blob_skel)[1]
            angle_half = 100
            
            # branches struct
            
            branch_struct = []
            
            branch_struct.append(Branch_Struct(branch_points_rows, branch_points_cols, thickness, sgl_blob, sgl_blob_skel))
            
            '''
            while angle_half>5 : 
                
                vector_1 = [branch_points_rows[0] - branch_points_rows[-1], branch_points_cols[0] - branch_points_cols[-1]]
                
                
              
                
                vector_1 = [branch_points_rows[-1], branch_points_cols[-1]]
                vector_2 = [branch_points_rows[0], branch_points_cols[0]]
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle_all = np.arccos(dot_product)
                
                branch_points_rows = np.array_split(branch_points_rows,2)[1]
                branch_points_cols = np.array_split(branch_points_cols,2)[1]
                
                
                vector_2 = [branch_points_rows[0] - branch_points_rows[-1], branch_points_cols[0] - branch_points_cols[-1]]
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle_half = np.arccos(dot_product)
                
                #branch_angle_diff = math.degrees(angle_all) - math.degrees(angle_half)
               
                
                
            plt.figure(4)
            plt.imshow(dst>0, cmap='gray')
            
            plt.figure(5)
            plt.imshow(blob_skeleton, cmap='gray')
        
            plt.figure(6)
            plt.imshow(sgl_blob, cmap='gray')
            
            plt.figure(7)
            plt.imshow(blob_img , cmap='gray')
        
            plt.show()
            '''
            
    
plt.figure(1)
plt.imshow(im_orig, cmap='gray')

plt.figure(2)
plt.imshow(masked_img_vas, cmap='gray')
plt.figure(3)
plt.imshow(masked_img, cmap='gray')
  
plt.show()

