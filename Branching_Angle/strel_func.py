'''
Created on 10 Jan 2021

@author: danan
'''
import numpy as np
def strel( line_length=15, degrees=0):
        
    deg90 = degrees%90
    if deg90 > 45:
        alpha = np.pi * (90 - deg90) / 180
    else:
        alpha = np.pi * deg90 / 180
    ray = (line_length - 1)/2;
     
    ## We are interested only in the discrete rectangle which contains the diameter
    ## However we focus our attention to the bottom left quarter of the circle,
    ## because of the central symmetry.
    c = int(round (ray * np.cos (alpha)) +1)
    r = int(round (ray * np.sin (alpha)) +1)
    ## Line rasterization
    line = np.zeros((r, c))
    m = np.tan(alpha)
    cols = np.array(range(1,c+1))
    rows = float(r) - np.fix (m * (cols - 0.5))
    for i in range(len(cols)):
        line[int(rows[i] - 1), int(cols[i] -1)] = 1
    #preparing blocks 
    linestrip = line[0,0:-1]
    linerest = line[1:,0:-1]
    z = np.zeros((r-1,c))
            
    #Assemblying blocks
    subA = np.hstack((z,linerest[::-1,::-1]))
    subB = np.hstack((linestrip,1,linestrip[::-1]))
    subC = np.hstack((linerest,z[::-1,::-1]))
    res = np.vstack((subA, subB, subC))
    
    #rotate transpose or flip
    sect = np.fix((degrees%180)/45)
    if sect == 1:
        #transpose res
        res = res.transpose()
    elif sect == 2:
        #90 deg rotation
        res = np.rot90(res)
    elif sect == 3:
        #fliplr
        res = np.fliplr(res)
    #otherwise do nothing
    
    return res



