'''
Created on 28 Feb 2021

@author: danan
'''
import numpy as np
import math

def measure_branch_angle(branch_1, branch_2, region_branch):
    
    #find centroid
    x_center, y_center = np.argwhere(region_branch==1).sum(0)/np.sum(region_branch)
    x_center = round(x_center)
    y_center = round(y_center)
    
    #find closest point to the branch
    min_distance = 300
    for i in range(len(branch_1.branch_points_rows)):
        distance = np.sqrt((branch_1.branch_points_rows[i]-x_center)**2 +(branch_1.branch_points_cols[i]-y_center)**2)
        
        if distance < min_distance:
            min_distance =distance
            closest_point_1 = [branch_1.branch_points_rows[i], branch_1.branch_points_cols[i]]
    
    min_distance = 300
    for i in range(len(branch_2.branch_points_rows)):
        distance = np.sqrt((branch_2.branch_points_rows[i]-x_center)**2 +(branch_2.branch_points_cols[i]-y_center)**2)
        
        if distance < min_distance:
            min_distance =distance
            closest_point_2 = [branch_2.branch_points_rows[i], branch_2.branch_points_cols[i]]
    
    # Calculate vectors - Branch 1
    branch_sgl_blob_copy = branch_1.sgl_blob_skel.copy()
    current_point = closest_point_1
    
    
    pixel_distance_angle_measure = 7
    j=0
    while j<pixel_distance_angle_measure and np.sum(branch_sgl_blob_copy)!=1:
        
        branch_sgl_blob_copy[current_point[0],current_point[1]] = 0
        new_point = np.where(branch_sgl_blob_copy[current_point[0]-1:current_point[0]+2,current_point[1]-1:current_point[1]+2])
        current_point = [current_point[0]+new_point[0][0]-1,current_point[1]+new_point[1][0]-1]
    
        j=j+1
    vector_1 = [current_point[0] - closest_point_1[0], current_point[1] - closest_point_1[1]]
    
    # Calculate vectors - Branch 2
    branch_sgl_blob_copy = branch_2.sgl_blob_skel.copy()
    current_point = closest_point_2
    
    j=0
    while j<pixel_distance_angle_measure and np.sum(branch_sgl_blob_copy)!=1:
        
        branch_sgl_blob_copy[current_point[0],current_point[1]] = 0
        new_point = np.where(branch_sgl_blob_copy[current_point[0]-1:current_point[0]+2,current_point[1]-1:current_point[1]+2])
        current_point = [current_point[0]+new_point[0][0]-1,current_point[1]+new_point[1][0]-1]
        
        j=j+1
    vector_2 = [current_point[0] - closest_point_2[0], current_point[1] - closest_point_2[1]]
    
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle_half = np.arccos(dot_product)
    
    return math.degrees(angle_half)
    