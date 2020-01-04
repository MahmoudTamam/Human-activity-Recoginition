import numpy as np
import pickle
import cv2
import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.preprocessing.image import ImageDataGenerator

class SF_DataGenerator(keras.utils.Sequence):
    def __init__(self, config, train_flag = False):
        'Initialization'
        self.config = config
        self.batch_size = self.config.batch_size
        self.n_classes = self.config.num_classes
        self.train_flag = train_flag

        if train_flag == True:
            data_pd = pd.read_pickle(self.config.train_csv)
        else:
            data_pd = pd.read_pickle(self.config.valid_csv)
        img_len = self.config.data_dir+'iiiiiiiimgs/train/c0/'+data_pd.iloc[0]['img']
        self.data = np.reshape(np.array([img_len, 'apples']), (1,2))
        self.data = np.repeat(self.data, len(data_pd), axis=0)

        for sf_data_itr in range(len(data_pd)):
            img_name = data_pd.iloc[sf_data_itr]['img']
            c_x = data_pd.iloc[sf_data_itr]['classname']
            self.data[sf_data_itr, :] = (self.config.data_dir+'imgs/train/'+c_x+'/'+img_name,(c_x.split('c')[1]))

        #Iterations and data length
        self.data_len = len(data_pd)
        self.data_iters = ( self.data_len + self.config.batch_size ) // self.config.batch_size

        #Keras Augmtations
        if train_flag == True:
            data_gen_args = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')
            self.augmentor =ImageDataGenerator(**data_gen_args)

    def __len__(self):
        return int(np.ceil(self.data_len / float(self.batch_size)))

    def on_epoch_end(self):
        if self.train_flag == True:
            np.random.shuffle(self.data)

    def __getitem__(self, idx):
        batch_start = idx*self.batch_size
        batch_end = (idx+1)*self.batch_size
        img_batch = []
        ann_batch = []

        for itr in range(batch_start, batch_end):
            if itr >= self.data_len:
                break
            img = cv2.imread(self.data[itr,0])
            img = cv2.resize(img, (self.config.resize[1], self.config.resize[0]))
            ann = keras.utils.to_categorical(int(self.data[itr,1]), num_classes=self.n_classes)
            if self.train_flag == True:
                img = self.augmentor.random_transform(img)
            img_batch.append(img)
            ann_batch.append(ann)
        img_batch = np.array(img_batch)
        ann_batch = np.array(ann_batch)
        return img_batch, ann_batch