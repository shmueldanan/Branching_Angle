
'''
Created on 14 Feb 2021

@author: danan
'''
import numpy as np
import ONH_Localization as ONHLOC
import ONH_Segmentation as ONHSEG
import measure_branch_angle as MSA
import Artery_Vein_Classify as AVC
import Tortuosity_calc as TC
import Arteriolar_narrowing as AN

import Create_angle_mask as cmask
import strel_func as strel
import math
from collections import namedtuple
import Map_tree as MT

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
    '''
    plt.figure(1)
    plt.imshow(blob_skeleton_stack, cmap='gray')
    
    plt.figure(2)
    plt.imshow(blob_img_stack, cmap='gray')
      
    plt.show()
    '''
    
    # Disconnected skeleton
    blob_region_skeleton = np.bitwise_xor(np.bitwise_and(blob_skeleton,blob_skeleton_bifurc_points>0),blob_skeleton)
    
    # Artery/Vein Classification
    art_vein_class,pixel_intense = AVC.Artery_Vein_Classify(blob_region_skeleton,blob_img,im_orig[:,:,2])
    '''
    plt.figure(1)
    plt.imshow(im_orig[:,:,1])
    
    plt.figure(2)
    plt.imshow(art_vein_class)
    
    plt.figure(3)
    plt.imshow(pixel_intense)
    
    plt.show()
    '''
    ret_skel, labels_skel = cv2.connectedComponents(np.uint8(blob_region_skeleton))
    regions_skel = regionprops(labels_skel)
    
    max_vess = 0
    
    print(region.label)
    
    if regions_skel.__len__()>1 and region.label >0:
    
    
        global branch_struct
        
        Branch_Struct = namedtuple("Branch_Struct", "label branch_points_rows branch_points_cols thickness sgl_blob sgl_blob_skel centroid orientation is_origin trunk_num trunk_branch")
        branch_struct = []

        print("Blob # " + str(region.label) + " Analysis:")
        
        
        for region_skel in regions_skel:
            
            if region_skel.area>3:
                sgl_blob_skel = labels_skel == region_skel.label
                selem = disk(4)
                sgl_blob_skel_dilate = dilation(sgl_blob_skel,selem)
                sgl_blob = np.bitwise_and(sgl_blob_skel_dilate,blob_img)
                
                thickness = np.sum(sgl_blob)/(np.sum(sgl_blob_skel))
                
                
                #region_skel.orientation()
                                    
                branch_points_rows=np.where(sgl_blob_skel)[0]
                branch_points_cols=np.where(sgl_blob_skel)[1]
                
                label = region.label
                # branches struct
                centroid = region_skel.centroid
                orientation = region_skel.orientation
                
                is_origin = 0
                trunk_num = 0
                trunk_branch = -1
                
                branch_struct.append(Branch_Struct(label,branch_points_rows, branch_points_cols, thickness, sgl_blob, sgl_blob_skel, centroid,orientation, is_origin,trunk_num , trunk_branch))
                
        
        # Mapping origin branches

        
        origin_num = []
        for k in range(len(branch_struct)):
            m=0
            while m < len(branch_struct[k].branch_points_rows):
                if math.sqrt(( branch_struct[k].branch_points_rows[m]-ONH_Center[1] )**2 + 
                                  ( branch_struct[k].branch_points_cols[m]-ONH_Center[0] )**2 ) <= ONH_Radius*2+5:
                    branch_struct[k] = branch_struct[k]._replace(is_origin=1)
                    
                    if not origin_num:
                        origin_num.append(k)
                    elif origin_num[-1] != k:
                        origin_num.append(k)
                    
                m=m+1
        
        
        # Go over all branching points in the current Blob
        Blob_branch_points = np.multiply(blob_skeleton_bifurc_points, blob_skeleton) > 1
        ret_branch, labels_branch = cv2.connectedComponents(np.uint8(Blob_branch_points))
        regions_branch = regionprops(labels_branch)
        
        branch_counter = 0
        for region_branch in regions_branch:
            branch_counter = branch_counter+1
            regions_branch_2 = regions_branch[branch_counter:]
            for region_branch_iter in regions_branch_2:
                if math.sqrt(( region_branch.centroid[0]-region_branch_iter.centroid[0] )**2 + 
                                  ( region_branch.centroid[1]-region_branch_iter.centroid[1] )**2 )<6:
                    
                    row_max = np.max(np.where(np.logical_or(labels_branch==region_branch.label,labels_branch==region_branch_iter.label))[0])
                    row_min = np.min(np.where(np.logical_or(labels_branch==region_branch.label,labels_branch==region_branch_iter.label))[0])    
                    col_max = np.max(np.where(np.logical_or(labels_branch==region_branch.label,labels_branch==region_branch_iter.label))[1])
                    col_min = np.min(np.where(np.logical_or(labels_branch==region_branch.label,labels_branch==region_branch_iter.label))[1])  
        
                    labels_branch[row_min:row_max+1,col_min:col_max+1] = region_branch.label
                    
                    #remove inside branches
                    m=0
                    while m<len(branch_struct):
                    
                        
                        branch_mult = np.multiply(branch_struct[m].sgl_blob_skel[row_min:row_max+1,col_min:col_max+1],labels_branch[row_min:row_max+1,col_min:col_max+1])
                        if np.any(branch_mult):
                            branch_struct.pop(m)
                        m=m+1
        
        
        
        
        global branch_pool
        
        branch_pool = list(range(len(branch_struct)))
        global labels_branch_copy
        
        trunk_branch =0 
        # Map based on origin trunks
        selem = disk(2)
        for k in range(len(origin_num)):
            
            labels_branch_copy = labels_branch.copy()
            
            if branch_pool:
                MT.Map_tree(origin_num[k], branch_struct, labels_branch_copy, branch_pool,k, trunk_branch)
            
            
        for l in range(len(branch_struct)):
            
            Tortuosity = TC.tortuosity_calc(branch_struct[l])
            average_thickness, minimal_thickness = AN.Arteriolar_narrowing(branch_struct[l])
            
            print("trunk_branch " + str(branch_struct[l].trunk_branch) + "  origin_num " + str(branch_struct[l].trunk_num) + "  Tortuosity: " + str(Tortuosity))
            
            
            
            
            
            '''
            plt.figure(1)
            plt.imshow(blob_skeleton_stack)
            plt.figure(2)
            plt.imshow(branch_struct[l].sgl_blob_skel, cmap='gray')
            plt.show()
            '''
            
        plt.figure(1)
        plt.imshow(labels_branch)
        
        plt.figure(2)
        plt.imshow(blob_img, cmap='gray')
        
        plt.figure(3)
        plt.imshow(blob_skeleton_stack, cmap='gray')
        
        plt.show()
        
        
                
        '''
        dilate_skel_blob = dilation(branch_struct[origin_num[k]][sgl_blob_skel],selem)
        connected_trunk_point = np.multiply(dilate_skel_blob,labels_branch_copy)
        branch_pool.remove(origin_num[k])
        
        # Classify in tree branch
        branch_struct[origin_num[k]][trunk_num] = k        
        branch_struct[origin_num[k]][trunk_branch] = 1
        
        if np.sum(connected_trunk_point)>0:
            connected_branches=[]
            for m in branch_pool:
                dilate_skel_blob = dilation(branch_struct[m][sgl_blob_skel],selem)
                connected_branch_point = np.multiply(dilate_skel_blob,labels_branch_copy)
                if np.sum(connected_branch_point)>0:
                    connected_branches.append(m)
                    
            
            # remove branch point
            labels_branch_copy[labels_branch_copy == connected_branch_point[np.where(connected_branch_point!=0)][0]] = 0
             
            # If branch       
            if len(connected_branches)==2:
                branch_struct[connected_branches[0]][trunk_num] = k
                branch_struct[connected_branches[1]][trunk_num] = k
                
                branch_struct[connected_branches[0]][trunk_branch] = 2
                branch_struct[connected_branches[1]][trunk_branch] = 2
                
                for n in range(2):
                
                    dilate_skel_blob = dilation(branch_struct[connected_branches[n]][sgl_blob_skel],selem)
                    connected_trunk_point = np.multiply(dilate_skel_blob,labels_branch_copy)
                    branch_pool.remove(connected_branches[n])
                    
                    if np.sum(connected_trunk_point)>0:
                        connected_branches_2=[]
                        for m in branch_pool:
                            dilate_skel_blob = dilation(branch_struct[m][sgl_blob_skel],selem)
                            connected_branch_point = np.multiply(dilate_skel_blob,labels_branch_copy)
                            if np.sum(connected_branch_point)>0:
                                connected_branches_2.append(m)
                                branch_pool.remove(m)
                        
            
            
                   
                    
            elif len(connected_branches)==1:
                branch_struct[connected_branches[0]][trunk_num] = k
                branch_struct[connected_branches[0]][trunk_branch] = 1
                
                
                
            elif len(connected_branches)==3: 
                #orientation
                
            '''
                
                
                
        
           
                   
    '''            
    for region_branch in regions_branch:
        
        
        branch_candidate = []
        for i in range(len(branch_struct)):
            j=0
            while j < len(branch_struct[i].branch_points_rows):
                
                
                if math.sqrt(( branch_struct[i].branch_points_rows[j]-region_branch.centroid[0] )**2 + 
                              ( branch_struct[i].branch_points_cols[j]-region_branch.centroid[1] )**2 ) < 3:
                    
                    j=10000
                    branch_candidate.append(i)
                    
                j=j+1
            
        if len(branch_candidate)==3:
            first_distance = math.sqrt((ONH_Center[0]-branch_struct[branch_candidate[0]].centroid[0] )**2 + (ONH_Center[1]-branch_struct[branch_candidate[0]].centroid[1] )**2)
            second_distance = math.sqrt((ONH_Center[0]-branch_struct[branch_candidate[1]].centroid[0] )**2 + (ONH_Center[1]-branch_struct[branch_candidate[1]].centroid[1] )**2)
            third_distance = math.sqrt((ONH_Center[0]-branch_struct[branch_candidate[2]].centroid[0] )**2 + (ONH_Center[1]-branch_struct[branch_candidate[2]].centroid[1] )**2)
            
            if first_distance<second_distance and first_distance<third_distance:
                Branching_angle = MSA.measure_branch_angle(branch_struct[branch_candidate[1]],branch_struct[branch_candidate[2]],region_branch)
            elif second_distance<third_distance:
                Branching_angle = MSA.measure_branch_angle(branch_struct[branch_candidate[0]],branch_struct[branch_candidate[2]],region_branch)
            else:
                Branching_angle = MSA.measure_branch_angle(branch_struct[branch_candidate[0]],branch_struct[branch_candidate[1]],region_branch)
             
        if len(branch_candidate)==3:
            if branch_struct[branch_candidate[0]].thickness>branch_struct[branch_candidate[1]].thickness and branch_struct[branch_candidate[0]].thickness>branch_struct[branch_candidate[2]].thickness:
                Branching_angle = MSA.measure_branch_angle(branch_struct[branch_candidate[1]],branch_struct[branch_candidate[2]],region_branch)
            elif branch_struct[branch_candidate[1]].thickness>branch_struct[branch_candidate[2]].thickness:
                Branching_angle = MSA.measure_branch_angle(branch_struct[branch_candidate[0]],branch_struct[branch_candidate[2]],region_branch)
            else:
                Branching_angle = MSA.measure_branch_angle(branch_struct[branch_candidate[0]],branch_struct[branch_candidate[1]],region_branch)
    
        
        print("Branching Angle:" + str(Branching_angle))        
    '''
plt.figure(1)
plt.imshow(im_orig, cmap='gray')

plt.figure(2)
plt.imshow(masked_img_vas, cmap='gray')
plt.figure(3)
plt.imshow(masked_img, cmap='gray')
  
plt.show()

