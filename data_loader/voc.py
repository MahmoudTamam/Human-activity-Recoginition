import numpy as np
import pickle
import cv2
from keras.preprocessing.image import ImageDataGenerator


list_single_label = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
]

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
        
        if self.config.single_label == True:
            filterd_train = []
            for train_ann in self.train_anns:
                label = train_ann['person'+str(0)+'action']
                if label in list_single_label:
                    filterd_train.append(train_ann)
            self.train_anns = filterd_train
            filterd_valid = []
            for valid_ann in self.valid_anns:
                label = valid_ann['person'+str(0)+'action']
                if label in list_single_label:
                    filterd_valid.append(valid_ann)
            self.valid_anns = filterd_valid
            for train_ann in self.train_anns:
                train_ann['person'+str(0)+'action'] = list_single_label.index(train_ann['person'+str(0)+'action'])
            for valid_ann in self.valid_anns:
                valid_ann['person'+str(0)+'action'] = list_single_label.index(valid_ann['person'+str(0)+'action'])
        #Iterations and data length
        self.train_len = len(self.train_anns)
        self.valid_len = len(self.valid_anns)
        self.train_iters = ( self.train_len + self.config.batch_size ) // self.config.batch_size
        self.valid_iters =  self.valid_len 
        #Augmentation
        data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
        self.augmentor =ImageDataGenerator(**data_gen_args)
        #Validation annotation
        self.valid_true = []
        for valid_ann in self.valid_anns:
            self.valid_true.append(valid_ann['person'+str(0)+'action'])
        self.valid_true = np.array(self.valid_true)
        self.train_true = []
        for tran_ann in self.train_anns:
            self.train_true.append(tran_ann['person'+str(0)+'action'])
        self.train_true = np.array(self.train_true)

    def train_generator(self):
        #TODO: Add multiple persons support
        while True:
            np.random.shuffle(self.train_anns)
            for batch_start in range(0, len(self.train_anns), self.config.batch_size):
                img_batch = []
                ann_batch = []
                for idx in range(batch_start, batch_start+self.config.batch_size):
                    person_idx = 0
                    if idx >= len(self.train_anns):
                        item =  self.train_anns[0]
                    else:
                        item =  self.train_anns[idx]
                    bbox = item['person'+str(person_idx)+'bbox']
                    img = cv2.imread(self.config.img_root+item['filename'])
                    if self.config.crop_images == True:
                        img = img[int(bbox[3]):int(bbox[2]), int(bbox[1]):int(bbox[0])]
                    img = cv2.resize(img, (self.config.resize[0], self.config.resize[1]))
                    img = self.augmentor.random_transform(img)
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
                img_batch = []
                ann_batch = []
                person_idx = 0
                img = cv2.imread(self.config.img_root+item['filename'])
                bbox = item['person'+str(person_idx)+'bbox']
                if self.config.crop_images == True:
                    img = img[int(bbox[3]):int(bbox[2]), int(bbox[1]):int(bbox[0])]
                img = cv2.resize(img, (self.config.resize[0], self.config.resize[1]))
                ann = item['person'+str(person_idx)+'action']
                img_batch.append(img)
                ann_batch.append(ann)
                img_batch = np.array(img_batch)
                ann_batch = np.array(ann_batch)
                yield img_batch, ann_batch