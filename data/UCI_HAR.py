import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))

features = list()
with open('UCI HAR Dataset/features.txt') as f:
    features = [line.split()[1] for line in f.readlines()]

train = pd.read_csv('UCI HAR Dataset/train/X_train.txt',sep='\s+', header=None, names=features)

y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', names=['Activity'], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})

train['Activity'] = y_train
train['ActivityName'] = y_train_labels

train.dropna()
train.drop_duplicates()

test = pd.read_csv('UCI HAR Dataset/test/X_test.txt',sep='\s+', header=None, names=features)

y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', names=['Activity'], squeeze=True)
y_test_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})

test['Activity'] = y_test
test['ActivityName'] = y_test_labels

test.dropna()
test.drop_duplicates()


plt.title('No of Datapoints per Activity', fontsize=15)
sns.countplot(train.ActivityName)
plt.xticks(rotation=90)
plt.show()