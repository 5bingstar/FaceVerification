#!/urs/bin/env python2


class AdaBoost:
    def __init__(self):
        self.features = []
        self.max_iterations = 5
        self.target_error = 0.01

        # data
        self.data = None
        self.actual = None

        # training weights
        self.weights = None

        # trained parameters
    
    def seed_features(self):
        if len(self.features) == 0:
            self.features = 
