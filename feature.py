#!/usr/env/bin python2

from config import *
from lbp import LBP

def get_lbp(File):
    file_path = data_path + File
    run = LBP(file_path)
    f = run.calculate_feature()
    return f

def get_feature():
    ppair, npair, test_pair = get_train_test()
    pfeatures = []
    for p in ppair[:10]: ##
        pf1 = get_lbp(p[0])
        pf2 = get_lbp(p[1])
        feature = [calculate_distance(pf1[i], pf2[i]) for i in range(len(pf1))]
        print feature[:10]
        pfeatures.append(feature)
    nfeatures = []
    for n in npair[:10]: ##
        nf1 = get_lbp(n[0])
        nf2 = get_lbp(n[1])
        feature = [calculate_distance(nf1[i], nf2[i]) for i in range(len(nf1))]
        print feature[:10]
        nfeatures.append(feature)
    test_features = []
    for p in test_pair[:10]: ## 
        pf1 = get_lbp(p[0])
        pf2 = get_lbp(p[1])
        feature = [calculate_distance(pf1[i], pf2[i]) for i in range(len(pf1))]
        test_features.append(feature)
    #print len(pfeatures)
    #print len(pfeatures[0])
    return pfeatures, nfeatures, test_features

def calculate_distance(x, y):
    if x == 0 and y == 0:
        return 0
    else:
        return pow(x-y, 2) / (x + y)

def get_train_test():
    ptrain_pair = []
    ntrain_pair = []
    test_pair = []
    f = open(train_name_path, 'r')
    content = [line.rstrip('\n') for line in f]
    f.close()
    for item in content[1: int(content[0])+1]:
        s = item.split()
        ptrain_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[0]+'/'+s[0]+'_'+s[2].zfill(4)+'.jpg'])
    for item in content[int(content[0])+1:]:
        s = item.split()
        ntrain_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[2]+'/'+s[2]+'_'+s[3].zfill(4)+'.jpg'])
    f = open(test_name_path, 'r')
    content = [line.rstrip('\n') for line in f]
    f.close()
    for item in content[1: int(content[0])+1]:
        s = item.split()
        test_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[0]+'/'+s[0]+'_'+s[2].zfill(4)+'.jpg'])
    for item in content[int(content[0])+1:]:
        s = item.split()
        test_pair.append([s[0]+'/'+s[0]+'_'+s[1].zfill(4)+'.jpg', s[2]+'/'+s[2]+'_'+s[3].zfill(4)+'.jpg'])
    return ptrain_pair, ntrain_pair, test_pair


