import numpy as np
import pandas as pd
import pickle

class VOCDataLoader():
    def __init__(self, config):
        self.config = config
        
        #Iterations and data length
        self.train_len = len(self.train_anns)
        self.valid_len = len(self.valid_anns)
        self.train_iters = ( self.train_len + self.config.batch_size ) // self.config.batch_size
        self.valid_iters =  self.valid_len 

    def train_generator(self):
        pass

    def valid_generator(self):
        pass