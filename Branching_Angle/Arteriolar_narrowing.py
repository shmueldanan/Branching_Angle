'''
Created on 18 Apr 2021

@author: danan
'''
import numpy as np

def Arteriolar_narrowing(branch_struct_current):
    
    #Find first point
    for k in range(np.sum(branch_struct_current.sgl_blob_skel)):
        if np.sum(branch_struct_current.sgl_blob_skel[branch_struct_current.branch_points_rows[k]-1:branch_struct_current.branch_points_rows[k]+2,
                                                      branch_struct_current.branch_points_cols[k]-1:branch_struct_current.branch_points_cols[k]+2])==2:
            current_point=[branch_struct_current.branch_points_rows[k], branch_struct_current.branch_points_cols[k]]
            
            
        
    
    branch_sgl_blob_copy = branch_struct_current.sgl_blob_skel.copy()
    j=0
    
    while np.sum(branch_sgl_blob_copy)!=2:
        
        branch_sgl_blob_copy[current_point[0],current_point[1]] = 0
        new_point = np.where(branch_sgl_blob_copy[current_point[0]-1:current_point[0]+2,current_point[1]-1:current_point[1]+2])
        current_point = [current_point[0]+new_point[0][0]-1,current_point[1]+new_point[1][0]-1]
    
        if j>=2 and j<np.sum(branch_struct_current.sgl_blob_skel)-2:
            
    