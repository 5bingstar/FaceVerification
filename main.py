#/usr/bin/env python2

import os 
import sys
from LBP import *

data_path = 'data/lfw-deepfunneled/'

if __name__ == "__main__":
    run = LBP(data_path + 'Zhang_Ziyi/Zhang_Ziyi_0001.jpg', 8, 2, 7, 5)
    run.execute()
    run.output_lbp()
