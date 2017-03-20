#!/usr/bin/env python2

import numpy as np
import math
import cv2
from skimage.feature import local_binary_pattern
#import matplotlib.pyplot as plt

class LBP:
    def __init__(self, input_file):
        self.image = cv2.imread(input_file.encode('gbk'), 0)
        h, w = self.image.shape
        self.image = self.image[h/2 - 25: h/2 + 25, w/2 - 25: w/2 + 25] # pick up 50*50 in center
        self.histogram = []
        
    def _lbp(self, count, radius):
        lbp = local_binary_pattern(self.image, count, radius, method = 'nri_uniform').astype(np.int32) # LBP feature for each pixel
        return lbp
    
    def _histogram(self, lbp, lBlock, rBlock, count):
        hCount = lBlock
        wCount = rBlock
        height, width = lbp.shape
        bin_num = count * (count - 1) + 3
        histogram = np.zeros((hCount*wCount, bin_num), dtype = np.float)  # one bins vector for each block
        for h in range(hCount):  
            for w in range(wCount):  
                blk = lbp[height*h/hCount: height*(h+1)/hCount, width*w/wCount: width*(w+1)/wCount]  
                hist1 = np.bincount(blk.ravel(), minlength = bin_num).astype(np.float) 
                histogram[h*wCount+w, :] = hist1 / hist1.sum() 
        return histogram.ravel() # combine to one vector

    def calculate_feature(self):
        s1 = [[8,2], [16,3], [16,4]]
        s2 = [[5,5], [4,4], [3,3], [2,2]]
        for i in s1:
            lbp = self._lbp(i[0], i[1])
            for j in s2:
                self.histogram += self._histogram(lbp, j[0], j[1], i[0]).tolist()
                #print type(self._histogram(lbp, j[0], j[1], i[0]))
        return self.histogram
'''
    def output_feature(self):
    def output_lbp(self):
        plt.subplot(133)
        plt.imshow(self.lbp, cmap='gray')
        plt.show()
'''
