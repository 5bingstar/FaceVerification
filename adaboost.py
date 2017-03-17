#!/urs/bin/env python2

import numpy as np
import math

class AdaBoost:
    def __init__(self, pdata, ndata, test_data):
        self.pfeatures = None
        self.nfeatures = None

        # data
        self.pdata = pdata
        self.ndata = ndata
        self.test_data = test_data

        #self.test_data = test_data
        self.T = 10
        self.D = 0

        # trained parameters
        self.alphas = []
        self.dimensions = []
        self.thresholds = []
        # tested 
        self.test_features = None
        self.test_results = []
    
    def seed_features(self):
        #if len(self.features) == 0:
        #    self.features = feature.get_feature(self.data)
        self.pfeatures = self.pdata
        self.nfeatures = self.ndata
        self.pnums, self.D = self.pfeatures.shape
        self.nnums, self.D = self.nfeatures.shape
    
    def weak_classifier(self, z, ru):
        if z <= ru:
            return 1
        else:
            return -1

    def _train_iteration(self):
        pweights = np.ones(self.pnums) / self.pnums
        nweights = np.ones(self.nnums) / self.nnums

        for i in range(self.T):
            dim, beta, ru = self.WeightedThresholdClassifier(pweights, nweights)
            self.dimensions.append(dim)
            self.thresholds.append(ru)
            for j in range(self.pnums):
                if self.pfeatures[j][dim] <= ru:
                    pweights[j] *= beta
            for j in range(self.nnums):
                if self.nfeatures[j][dim] > ru:
                    nweights[j] *= beta
            if beta == 0:
                self.alphas.append(10)
            else:
                self.alphas.append(math.log(1/beta))
    
    def _test_iteration(self):
        #self.test_features = feature.get_features(self.test_data)
        self.test_features = self.test_data
        for item in self.test_features:
            score = 0
            for i in range(self.T):
                score += self.alphas[i] * self.weak_classifier(self.dimensions[i], self.thresholds[i])
            if score > 0:
                self.test_results.append(1)
            else:
                self.test_results.append(-1)
        print self.test_results
        
    def WeightedThresholdClassifier(self, pweights, nweights):
        dim = 0
        beta = 0
        ru = 0
        min_eps = 1
        for i in range(self.D):
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
            if Max1 <= Min0:
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
        print dim, beta, ru
        return dim, beta, ru

