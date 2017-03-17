#!/usr/env/bin python2

from config import *
#from lbp import LBP

def get_lbp(file_path):
    for picture in data:
        LBP p(picture, 8, 2, 10, 10)
        p.execute()
        f = p.output_feature()
        features.append(f)
    return features

def get_feature():
    ppair, npair, test_pair = get_train_test()
    pfeatures = []
    for p in ppair:
        pf1 = get_lbp(p[0])
        pf2 = get_lbp(p[1])
        feature = [calculate_distance(pf1[i], pf2[i]) for i in range(len(pf1))]
        pfeatures.append(feature)
    nfeatures = []
    for n in npair:
        nf1 = get_lbp(n[0])
        nf2 = get_lbp(n[1])
        feature = [calculate_distance(nf1[i], nf2[i]) for i in range(len(nf1))]
        nfeatures.append(feature)
    test_features = []
    for p in test_pair:
        pf1 = get_lbp(p[0])
        pf2 = get_lbp(p[1])
        feature = [calculate_distance(pf1[i], pf2[i]) for i in range(len(pf1))]
        test_features.append(feature)
    return pfeatures, nfeatures, test_features

def calculate_distance(x, y):
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
        ptrain_pair.append([s[0]+'_'+s[1]+'.jpg', s[0]+'_'+s[2]+'.jpg'])
    for item in content[int(content[0])+1:]:
        s = item.split()
        ntrain_pair.append([s[0]+'_'+s[1]+'.jpg', s[2]+'_'+s[3]+'.jpg'])
    f = open(test_name_path, 'r')
    content = [line.rstrip('\n') for line in f]
    f.close()
    for item in content[1: int(content[0])+1]:
        s = item.split()
        test_pair.append([s[0]+'_'+s[1]+'.jpg', s[0]+'_'+s[2]+'.jpg'])
    for item in content[int(content[0])+1:]:
        s = item.split()
        test_pair.append([s[0]+'_'+s[1]+'.jpg', s[2]+'_'+s[3]+'.jpg'])
    return ptrain_pair, ntrain_pair, test_pair


