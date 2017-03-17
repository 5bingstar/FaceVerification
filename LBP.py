#!/usr/bin/env python2

import numpy as np
import math
import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

class LBP:
    def __init__(self, input_file, count, radius, lBlock, rBlock):
        self.image = cv2.imread(input_file.encode('gbk'), 0)
        h, w = self.image.shape
        #self.image = self.image[h/2 - 25: h/2 + 25, w/2 - 25: w/2 + 25] # pick up 50*50 in center
        self.count = count
        self.radius = radius
        self.lBlock = lBlock
        self.rBlock = rBlock

    def execute(self):
        self._lbp()
        self._histogram()
        
    def _lbp(self):
        self.lbp = local_binary_pattern(self.image, self.count, self.radius, method = 'nri_uniform').astype(np.int32) # LBP feature for each pixel
        '''
        dh = np.round([self.radius * math.sin(i * 2 * math.pi / self.count) for i in range(self.count)])  
        dw = np.round([self.radius * math.cos(i * 2 * math.pi / self.count) for i in range(self.count)])  
        height ,width = self.image.shape  
        self.lbp = np.zeros(self.image.shape, dtype = np.int)  
        print self.image
        I1 = np.pad(self.image, self.radius, 'edge')  
        for k in range(self.count):  
            h,w = self.radius + int(dh[k]), self.radius + int(dw[k])  
            self.lbp += ((self.image>I1[h:h+height,w:w+width])<<k)  
        '''
    def _histogram(self):
        lbp = self.lbp
        hCount = self.lBlock
        wCount = self.rBlock
        height, width = lbp.shape
        bin_num = self.count * (self.count - 1) + 3
        self.histogram = np.zeros((hCount*wCount, bin_num), dtype = np.float)  # one bins vector for each block
        for h in range(hCount):  
            for w in range(wCount):  
                blk = lbp[height*h/hCount: height*(h+1)/hCount, width*w/wCount: width*(w+1)/wCount]  
                hist1 = np.bincount(blk.ravel(), minlength = bin_num).astype(np.float) 
                self.histogram[h*wCount+w, :] = hist1 / hist1.sum() 
    
    def output_lbp(self):
        plt.subplot(133)
        plt.imshow(self.lbp, cmap='gray')
        plt.show()

    def output_feature(self):
        print self.histogram.ravel() # combine to one vector
