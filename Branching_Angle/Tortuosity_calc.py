'''
Created on 18 Apr 2021

@author: danan
'''
import numpy as np

def tortuosity_calc(branch_struct_current):
    
    chord_len = np.sqrt((branch_struct_current.branch_points_rows[0] - branch_struct_current.branch_points_rows[-1])**2+(branch_struct_current.branch_points_cols[0] - branch_struct_current.branch_points_cols[-1])**2)
    
    arc_len = 1
    for line_ind in range(len(branch_struct_current.branch_points_rows)-1):
        if branch_struct_current.branch_points_rows[line_ind]==branch_struct_current.branch_points_rows[line_ind+1] or branch_struct_current.branch_points_cols[line_ind]==branch_struct_current.branch_points_cols[line_ind+1]:
            arc_len = arc_len +1
        else:
            arc_len = arc_len + 1.41
    
    return arc_len/chord_len