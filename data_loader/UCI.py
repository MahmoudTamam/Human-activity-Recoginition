import numpy as np
import pandas as pd
import pickle
import keras

class UCIDataLoader():
    def __init__(self, config):
        self.config = config
        #load data #TODO: add into configuration
        if self.config.model == 'LSTM' or self.config.model == 'CNN' or self.config.model == 'CNN_LSTM':
            self.X_train = np.float32(np.load("data/inertial_X_train.npy"))
            self.Y_train = np.int_(np.load("data/inertial_Y_train.npy"))-1
            self.X_valid = np.float32(np.load("data/inertial_X_valid.npy"))
            self.Y_valid = np.int_(np.load("data/inertial_Y_valid.npy"))-1   
            self.X_test = np.float32(np.load("data/inertial_X_test.npy"))
            self.Y_test = np.int_(np.load("data/inertial_Y_test.npy"))-1
        elif self.config.model == 'FC':
            if self.config.PCA_applied :           
                self.X_train = np.float32(np.load("data/X_trainPca.npy"))
                self.Y_train = np.int_(np.load("data/Y_trainPca.npy"))-1
                self.X_valid = np.float32(np.load("data/X_validPca.npy"))
                self.Y_valid = np.int_(np.load("data/Y_validPca.npy"))-1   
                self.X_test = np.float32(np.load("data/X_testPca.npy"))
                self.Y_test = np.int_(np.load("data/Y_testPca.npy"))-1
            else :
                self.X_train = np.float32(np.load("data/X_train.npy"))
                self.Y_train = np.int_(np.load("data/Y_train.npy"))-1
                self.X_valid = np.float32(np.load("data/X_valid.npy"))
                self.Y_valid = np.int_(np.load("data/Y_valid.npy"))-1   
                self.X_test = np.float32(np.load("data/X_test.npy"))
                self.Y_test = np.int_(np.load("data/Y_test.npy"))-1
                
        #Iterations and data length
        self.train_len = len(self.Y_train)
        self.valid_len = len(self.Y_valid)
        self.test_len = len(self.Y_test)
        #Category
        #self.Y_train = keras.utils.to_categorical(self.Y_train, num_classes=self.config.classes_num)
        #self.Y_valid = keras.utils.to_categorical(self.Y_valid, num_classes=self.config.classes_num)
        #self.Y_test = keras.utils.to_categorical(self.Y_test, num_classes=self.config.classes_num)

        self.train_iters = ( self.train_len + self.config.batch_size ) // self.config.batch_size
        self.valid_iters = ( self.valid_len + self.config.batch_size ) // self.config.batch_size
        self.test_iters = ( self.test_len + self.config.batch_size ) // self.config.batch_size

        #TODO: Normalization of features [0,1] or [-1,1] #Update: Data already normalized
