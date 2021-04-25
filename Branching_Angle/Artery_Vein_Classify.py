'''
Created on 14 Mar 2021

@author: danan
'''
import numpy as np

def Artery_Vein_Classify(blob_region_skeleton,blob_img,masked_img):
    
    art_vein_class = np.zeros(blob_img.shape)
    pixel_intense = np.zeros(blob_img.shape)
    
    indexes = np.where(blob_region_skeleton)
    
    for i in range(len(indexes[0])):
        five_box = blob_img[indexes[0][i]-2:indexes[0][i]+3,indexes[1][i]-2:indexes[1][i]+3]
        ten_box = 1 - blob_img[indexes[0][i]-5:indexes[0][i]+6,indexes[1][i]-5:indexes[1][i]+6]
        
        AVR_ratio = np.sum(np.multiply(masked_img[indexes[0][i]-2:indexes[0][i]+3,indexes[1][i]-2:indexes[1][i]+3],five_box))/np.sum(np.multiply(masked_img[indexes[0][i]-5:indexes[0][i]+6,indexes[1][i]-5:indexes[1][i]+6],ten_box))
                    
        art_vein_class[indexes[0][i],indexes[1][i]] = AVR_ratio
        pixel_intense[indexes[0][i],indexes[1][i]] = masked_img[indexes[0][i],indexes[1][i]]
        
    return art_vein_class,pixel_intense