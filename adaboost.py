#!/urs/bin/env python2

import os
import numpy as np
import math
from config import *

class AdaBoost:
    def __init__(self):
        self.pfeatures = None
        self.nfeatures = None
        self.test_features = None
        # training setting
        self.T = 150
        self.D = None
        self.pnums = 0
        self.nnums = 0
        # trained parameters
        self.alphas = []
        self.dimensions = []
        self.thresholds = []
        # tested result
        self.test_results = []
    
    def seed_features(self):
        if os.path.exists(data_path + 'pfeature.csv'):
            self.pfeatures = np.loadtxt(data_path + 'pfeature.csv', delimiter = ',', skiprows=0).tolist()
        if os.path.exists(data_path + 'nfeature.csv'):
            self.nfeatures = np.loadtxt(data_path + 'nfeature.csv', delimiter = ',', skiprows=0).tolist()
        else:
            import feature
            self.pfeatures, self.nfeatures = feature.get_train_features()
            np.savetxt(data_path + 'pfeature.csv', self.pfeatures, delimiter = ',')
            np.savetxt(data_path + 'nfeature.csv', self.nfeatures, delimiter = ',')

        self.pnums = len(self.pfeatures)
        self.D = len(self.pfeatures[0])
        self.nnums = len(self.nfeatures)
    
    def weak_classifier(self, z, ru):
        if z <= ru:
            return 1
        else:
            return -1

    def _train_iteration(self):
        pweights = np.ones(self.pnums) / self.pnums
        nweights = np.ones(self.nnums) / self.nnums

        for i in range(self.T):
            print "iteration %d:" %(i+1)
            dim, beta, ru = self.WeightedThresholdClassifier(pweights, nweights)
            self.dimensions.append(dim)
            self.thresholds.append(ru)
            print "choose dim=%d" %dim
            print "choose ru=%f" %ru
            for j in range(self.pnums):
                if self.pfeatures[j][dim] <= ru:
                    pweights[j] *= beta
            for j in range(self.nnums):
                if self.nfeatures[j][dim] > ru:
                    nweights[j] *= beta
            normalize = sum(pweights) + sum(nweights)
            pweights = [x/normalize for x in pweights]
            nweights = [x/normalize for x in nweights]
            if beta == 0:
                self.alphas.append(10)
            else:
                self.alphas.append(math.log(1/beta))
            print "choose alpha=%f" %self.alphas[-1]
    
    def _test_iteration(self):
        if os.path.exists(data_path + 'test_feature.csv'):
            self.test_features = np.loadtxt(data_path + 'test_feature.csv', delimiter = ',', skiprows=0).tolist()
        else:
            import feature
            self.test_features = feature.get_test_features()
            np.savetxt(data_path + 'test_feature.csv', self.test_features, delimiter = ',')
        for item in self.test_features:
            score = 0
            for i in range(self.T):
                score += self.alphas[i] * self.weak_classifier(item[self.dimensions[i]], self.thresholds[i])
            if score > 0:
                self.test_results.append('1')
            else:
                self.test_results.append('-1')
        return self.test_results

    def _save_result(self):
        with open('result_all', 'w') as fout:
            fout.write('\n'.join(self.test_results))
        
    def WeightedThresholdClassifier(self, pweights, nweights):
        dim = 0
        beta = 0
        ru = 0
        min_error = 1
        for i in range(self.D):
            print "process in dimension %d" %(i+1)
            pvalue = [[t, self.pfeatures[t][i]] for t in range(self.pnums)]
            nvalue = [[t, self.nfeatures[t][i]] for t in range(self.nnums)]
            ppair = sorted(pvalue, key = lambda x : x[1])
            npair = sorted(nvalue, key = lambda x : x[1])
            Max1 = ppair[-1][1]
            Min0 = npair[0][1]
            if Max1 < Min0:
                return i, 0, Max1
            site1 = 0
            for t in range(self.pnums):
                if ppair[t][1] >= Min0:
                    site1 = t
                    break
            site2 = 0
            for t in range(self.nnums):
                if npair[t][1] <= Max1:
                    site2 = t
                else:
                    break
            total_error = 0
            for t in range(site1, self.pnums):
                total_error += pweights[ppair[t][0]]
            error = total_error
            start = 0
            for t in range(site1, self.pnums):
                r = ppair[t][1]
                error -= pweights[ppair[t][0]] 
                for j in range(start, site2+1):
                    if npair[j][1] <= r:
                        error += nweights[npair[j][0]]
                    else:
                        start = j
                        break
                if error < min_error:
                    min_error = error
                    dim = i
                    ru = r
                    beta = error / (1-error)
            error = total_error
            start = site1
            for t in range(site2):
                r = npair[t][1]
                error += nweights[npair[t][0]] 
                for j in range(start, self.pnums):
                    if ppair[j][1] < r:
                        error -= pweights[ppair[j][0]]
                    else:
                        start = j
                        break
                if error < min_error:
                    min_error = error
                    dim = i
                    ru = r
                    beta = error / (1-error)
        return dim, beta, ru

        '''
        dim = 0
        beta = 0
        ru = 0
        min_eps = 1
        for i in range(3):#self.D):
            print "process in dimension %d" %(i+1)
            Min1 = 1
            Max1 = 0
            Min0 = 1
            Max0 = 0
            for t in range(self.pnums):
                if self.pfeatures[t][i] < Min1:
                    Min1 = self.pfeatures[t][i]
                if self.pfeatures[t][i] > Max1:
                    Max1 = self.pfeatures[t][i]
            for t in range(self.nnums):
                if self.nfeatures[t][i] < Min0:
                    Min0 = self.nfeatures[t][i]
                if self.nfeatures[t][i] > Max0:
                    Max0 = self.nfeatures[t][i]
            if Max1 < Min0:
                return i, 0, Max1
            else:
                possible_ru = []
                for t in range(self.pnums):
                    s = self.pfeatures[t][i]
                    if s >= Min0 and s <= Max1:
                        if s not in possible_ru:
                            possible_ru.append(s)
                for t in range(self.nnums):
                    s = self.nfeatures[t][i]
                    if s >= Min0 and s <= Max1:
                        if s not in possible_ru:
                            possible_ru.append(s)
                for r in possible_ru:
                    eps = 0
                    for t in range(self.pnums):
                        eps += pweights[t] * (1 - self.weak_classifier(self.pfeatures[t][i], r))
                    for t in range(self.nnums):
                        eps += nweights[t] * (1 + self.weak_classifier(self.nfeatures[t][i], r))
                    eps /= 2
                    if eps < min_eps:
                        min_eps = eps
                        ru = r
                        dim = i
                        beta = eps / (1 - eps)
        print "min_eps = %f" %min_eps
        print "*dim = %d" %dim
        print "*beta = %d" %beta
        print "*ru = %d" %ru
        #return dim, beta, ru
        min_error = 1
        for i in range(self.D):
            print "process in dimension %d" %(i+1)
            pvalue = [[t, self.pfeatures[t][i]] for t in range(self.pnums)]
            nvalue = [[t, self.nfeatures[t][i]] for t in range(self.nnums)]
            ppair = sorted(pvalue, key = lambda x : x[1])
            npair = sorted(nvalue, key = lambda x : x[1])
            Max1 = ppair[-1][1]
            Min0 = npair[0][1]
            if Max1 < Min0:
                return i, 0, Max1
            site1 = 0
            for t in range(self.pnums):
                if ppair[t][1] >= Min0:
                    site1 = t
                    break
            site2 = 0
            for t in range(self.nnums):
                if npair[t][1] <= Max1:
                    site2 = t
                else:
                    break
            p = site1
            n = 0
            while(p < self.pnums and n <= site2):
                if ppair[p][1] == npair[n][1]:
                    r = ppair[p][1]
                    error = 0
                    for j in range(p+1, self.pnums):
                        error += pweights[ppair[j][0]]
                    for j in range(n+1):
                        error += nweights[npair[j][0]]
                    p += 1
                    n += 1
                    if error < min_error:
                        min_error = error
                        dim = i
                        ru = r
                        beta = error / (1-error)

                elif ppair[p][1] < npair[n][1]:
                    r = ppair[p][1]
                    error = 0
                    for j in range(p+1, self.pnums):
                        error += pweights[ppair[j][0]]
                    for j in range(n):
                        error += nweights[npair[j][0]]
                    p += 1
                    if error < min_error:
                        min_error = error
                        dim = i
                        ru = r
                        beta = error / (1-error)

                else:
                    r = ppair[n][1]
                    error = 0
                    for j in range(p+1, self.pnums):
                        error += pweights[ppair[j][0]]
                    for j in range(n):
                        error += nweights[npair[j][0]]
                    n += 1
                    if error < min_error:
                        min_error = error
                        dim = i
                        ru = r
                        beta = error / (1-error)
        return dim, beta, ru
         min_error = 1
      '''
