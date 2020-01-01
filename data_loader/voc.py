import numpy as np
import pickle
import cv2
from keras.preprocessing.image import ImageDataGenerator

class VOCDataLoader():
    def __init__(self, config):
        self.config = config
        self.train_anns = pickle.load(open(self.config.train_pickle,"rb"))
        self.valid_anns = pickle.load(open(self.config.valid_pickle,"rb"))
        #Filter data
        filterd_train = []
        for train_ann in self.train_anns:
            if train_ann['num_persons'] <= self.config.person_num:
                filterd_train.append(train_ann)
        self.train_anns = filterd_train
        filterd_valid = []
        for valid_ann in self.valid_anns:
            if valid_ann['num_persons'] <= self.config.person_num:
                filterd_valid.append(valid_ann)
        self.valid_anns = filterd_valid
        #Iterations and data length
        self.train_len = len(self.train_anns)
        self.valid_len = len(self.valid_anns)
        self.train_iters = ( self.train_len + self.config.batch_size ) // self.config.batch_size
        self.valid_iters =  self.valid_len 
        #Augmentation
        #TODO: Add Augmentation flow
        data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    def train_generator(self):
        #TODO: Add multiple persons support
        while True:
            np.random.shuffle(self.train_anns)
            for batch_start in range(0, len(self.train_anns), self.config.batch_size):    
                img_batch = []
                ann_batch = []
                for idx in range(batch_start, batch_start+self.config.batch_size):
                    person_idx = 0
                    if batch_start+self.config.batch_size > len(self.train_anns):
                        break
                    item =  self.train_anns[idx]
                    img = cv2.imread(self.config.img_root+item['filename'])
                    mask = np.zeros(img.shape)[:,:,0]
                    #num_persons = item['num_persons']
                    #for idx in range(num_persons):#TODO:
                    #bbox = item['person'+str(person_idx)+'bbox']
                    #mask = cv2.rectangle(mask, (int(bbox[1]),int(bbox[3])), (int(bbox[0]),int(bbox[2])), 255,-1)
                    #if self.config.batch_size > 1:#Resize batch
                    img = cv2.resize(img, (self.config.resize[0], self.config.resize[1]))
                    #mask = cv2.resize(mask, (self.config.resize[0], self.config.resize[1]))
                    #img = np.concatenate((img, mask), axis=2)
                    ann = item['person'+str(person_idx)+'action']
                    img_batch.append(img)
                    ann_batch.append(ann)
                img_batch = np.array(img_batch)
                ann_batch = np.array(ann_batch)
                yield img_batch, ann_batch

    def valid_generator(self):
        while True:
            np.random.shuffle(self.valid_anns)
            for item in self.valid_anns:
                img = cv2.imread(self.config.img_root+item['filename'])
                #mask = np.zeros(img.shape)[:,:,0]
                num_persons = item['num_persons']
                for idx in range(num_persons):
                    #bbox = item['person'+str(idx)+'bbox']
                    #mask = cv2.rectangle(mask, (bbox[1],bbox[3]), (bbox[0],bbox[2]), 255,-1)
                    #if self.config.batch_size > 1:#Resize batch
                    img = cv2.resize(img, (self.config.resize[0], self.config.resize[1]))
                    #mask = cv2.resize(mask, (self.config.resize[0], self.config.resize[1]))
                    #img = np.concatenate((img, mask), axis=2)
                    ann = item['person'+str(idx)+'action']
                yield img, ann