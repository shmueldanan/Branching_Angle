'''
Created on 21 Mar 2021

@author: danan
'''
from skimage.morphology import erosion,dilation,disk
import numpy as np
import matplotlib.pyplot as plt 
import measure_branch_angle as MSA


def Map_tree(start_branch, branch_struct, labels_branch_copy, branch_pool,k, trunk_branch):
    selem = disk(2)

    dilate_skel_blob = dilation(branch_struct[start_branch].sgl_blob_skel,selem)
    connected_trunk_point = np.multiply(dilate_skel_blob,labels_branch_copy)
    
    
    # Classify in tree branch
    branch_struct[start_branch] = branch_struct[start_branch]._replace(trunk_num=k)
    branch_struct[start_branch] = branch_struct[start_branch]._replace(trunk_branch=trunk_branch)
    
    # Remove current branch
    if start_branch in branch_pool:
        branch_pool.remove(start_branch)
    if len(branch_pool)==0:
        return
    
    connected_branches=[]
    if np.sum(connected_trunk_point)>0:
        
        current_label = labels_branch_copy == connected_trunk_point[np.where(connected_trunk_point!=0)][0]
        for m in branch_pool:
            dilate_skel_blob = dilation(branch_struct[m].sgl_blob_skel,selem)
            connected_branch_point = np.multiply(dilate_skel_blob,current_label)
            if np.sum(connected_branch_point)>0:
                connected_branches.append(m)
                
        
        # remove branch point
        labels_branch_copy[labels_branch_copy == labels_branch_copy[np.where(current_label!=0)[0][0],np.where(current_label!=0)[1][0]]] = 0
    
    # If branch
    if len(connected_branches)==2:
        
        # Measure branching angle
        Branching_angle = MSA.measure_branch_angle(branch_struct[connected_branches[0]],branch_struct[connected_branches[1]],current_label)
        Optimality_ratio = (branch_struct[connected_branches[0]].thickness + branch_struct[connected_branches[1]].thickness)/(2*branch_struct[start_branch].thickness)
                
       
        
        if Branching_angle>130:
            return
        
        else:
            print("Branching angle: " + str(Branching_angle) + "   Optimality Ratio: " + str(Optimality_ratio) + "  Origin: " + str(branch_struct[start_branch].trunk_num) + "  Branch: " + str(branch_struct[start_branch].trunk_branch))
            
            
            for n in range(2):
                Map_tree(connected_branches[n], branch_struct, labels_branch_copy, branch_pool, k , trunk_branch+1) 
            
    elif len(connected_branches)==1:
        Map_tree(connected_branches[0], branch_struct, labels_branch_copy, branch_pool, k , trunk_branch) 
           
    elif len(connected_branches)==3:
        orient_dis = 1000
        for num_branch in range(len(connected_branches)):
            if np.abs(branch_struct[connected_branches[num_branch]].orientation-branch_struct[start_branch].orientation)<orient_dis:
                selected_branch = connected_branches[num_branch]
                orient_dis = np.abs(branch_struct[connected_branches[num_branch]].orientation-branch_struct[start_branch].orientation)
                
        Map_tree(selected_branch, branch_struct, labels_branch_copy, branch_pool, k , trunk_branch) 
        
        
    else:
        return
    