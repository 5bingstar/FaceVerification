#/usr/bin/env python2

import os 
import sys
from adaboost import AdaBoost

if __name__ == "__main__":
    run = AdaBoost()
    run.seed_features()
    run._train_iteration()
    run._test_iteration()
    run._save_result()
