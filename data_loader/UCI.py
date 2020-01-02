import numpy as np
import pandas as pd
import pickle

class UCIDataLoader():
    def __init__(self, config):
        self.config = config
        #load data #TODO: add into configuration
        self.X_train = np.float32(np.load("data/X_train.npy"))
        self.Y_train = np.float32(np.load("data/Y_train.npy"))
        self.X_valid = np.float32(np.load("data/X_valid.npy"))
        self.Y_valid = np.float32(np.load("data/Y_valid.npy"))   
        self.X_test = np.float32(np.load("data/X_test.npy"))
        self.Y_test = np.float32(np.load("data/Y_test.npy"))
        #Iterations and data length
        self.train_len = len(self.Y_train)
        self.valid_len = len(self.Y_valid)
        self.test_len = len(self.Y_test)

        self.train_iters = ( self.train_len + self.config.batch_size ) // self.config.batch_size
        self.valid_iters = ( self.valid_len + self.config.batch_size ) // self.config.batch_size
        self.test_iters = ( self.test_len + self.config.batch_size ) // self.config.batch_size

        #TODO: Normalization of features [0,1] or [-1,1]