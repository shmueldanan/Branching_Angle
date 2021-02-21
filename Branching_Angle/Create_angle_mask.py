'''
Created on 18 Oct 2020

@author: danan
'''

import numpy as np
import math


def create_circular_mask(h, w, center=None, radius=None, radius_small=None, theta = None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    mask = dist_from_center <= radius  
    mask2 = dist_from_center >= radius_small
    
    mask = mask&mask2
    
    inside_angles = np.arctan(np.divide(X- center[0],Y-center[1]+0.000001))
    inside_angles[0:center[1],:] = inside_angles[0:center[1],:]-1.57
    inside_angles[center[1]:,:] = inside_angles[center[1]:,:]+1.57
    
    
    return mask