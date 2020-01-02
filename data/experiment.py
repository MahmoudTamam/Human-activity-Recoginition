import numpy as np
import pickle
import cv2


train_anns = pickle.load(open("train_data.pickle","rb"))

counter_pos = 0
counter_neg = 0

list_pos = [
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

filterd_train = []
for train_ann in train_anns:
    if train_ann['num_persons'] <= 1:
        filterd_train.append(train_ann)
train_anns = filterd_train

filtered = []
for train_ann in train_anns:
    label = train_ann['person'+str(0)+'action']
    if label in list_pos:
        counter_pos+=1
        filtered.append(train_ann)
    else:
        counter_neg+=1
train_anns = filtered

for train_ann in train_anns:
    train_ann['person'+str(0)+'action'] = list_pos.index(train_ann['person'+str(0)+'action'])

train_true = []
for tran_ann in train_anns:
    train_true.append(tran_ann['person'+str(0)+'action'])
train_true = np.array(train_true)

print(len(train_anns))
print(counter_pos)
print(counter_neg)

print(train_true.shape)
print(np.unique(train_true, axis = 0, return_counts = True))
exit(0)