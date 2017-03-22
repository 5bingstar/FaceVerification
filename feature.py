#!/usr/env/bin python2

from config import *
from lbp import LBP

def get_lbp(File):
    file_path = data_path + File
    run = LBP(file_path)
    f = run.calculate_feature()
    return f

def calculate_distance(x, y):
    if x == 0 and y == 0:
        return 0
    else:
        return pow(x-y, 2) / (x + y)

def get_train_features():
    ptrain_pair = []
    ntrain_pair = []
    f = open(train_name_path, 'r')
    content = [line.rstrip('\n') for line in f]
    f.close()
    for item in content[1: int(content[0])+1]:
        s = item.split()
        ptrain_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[0]+'/'+s[0]+'_'+s[2].zfill(4)+'.jpg'])
    for item in content[int(content[0])+1:]:
        s = item.split()
        ntrain_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[2]+'/'+s[2]+'_'+s[3].zfill(4)+'.jpg'])
    print "Prepare positive features..."
    pfeatures = []
    for p in ppair: ##
        pf1 = get_lbp(p[0])
        pf2 = get_lbp(p[1])
        feature = [calculate_distance(pf1[i], pf2[i]) for i in range(len(pf1))]
        pfeatures.append(feature)
    print "Prepare negative features..."
    nfeatures = []
    for n in npair: ##
        nf1 = get_lbp(n[0])
        nf2 = get_lbp(n[1])
        feature = [calculate_distance(nf1[i], nf2[i]) for i in range(len(nf1))]
        nfeatures.append(feature)
    print "Prepare testing features..."
    return pfeatures, nfeatures

def get_test_features():
    test_pair = []
    f = open(test_name_path, 'r')
    content = [line.rstrip('\n') for line in f]
    f.close()
    for item in content[1: int(content[0])+1]:
        s = item.split()
        test_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[0]+'/'+s[0]+'_'+s[2].zfill(4)+'.jpg'])
    for item in content[int(content[0])+1:]:
        s = item.split()
        test_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[2]+'/'+s[2]+'_'+s[3].zfill(4)+'.jpg'])
    print "Prepare testing features..."
    test_features = []
    for p in test_pair: ## 
        pf1 = get_lbp(p[0])
        pf2 = get_lbp(p[1])
        feature = [calculate_distance(pf1[i], pf2[i]) for i in range(len(pf1))]
        test_features.append(feature)
    return test_features
