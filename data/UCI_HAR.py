import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

features = list()
with open('UCI HAR Dataset/features.txt') as f:
    features = [line for line in f.readlines()]

train = pd.read_csv('UCI HAR Dataset/train/X_train.txt',sep='\s+', header=None, names=features)

y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', names=['Activity'], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})

train.dropna()
train.drop_duplicates()

X_train, X_valid, Y_train, Y_valid = train_test_split(train, y_train, test_size=0.2, random_state=42)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_valid.npy", X_valid)
np.save("Y_valid.npy", Y_valid)

test = pd.read_csv('UCI HAR Dataset/test/X_test.txt',sep='\s+', header=None, names=features)

y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', names=['Activity'], squeeze=True)
y_test_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})

test.dropna()
test.drop_duplicates()


np.save("X_test.npy", test)
np.save("Y_test.npy", y_test)



train['Activity'] = y_train
train['ActivityName'] = y_train_labels
test['Activity'] = y_test
test['ActivityName'] = y_test_labels

plt.title('No of Datapoints per Activity', fontsize=15)
sns.countplot(train.ActivityName)
plt.xticks(rotation=90)
plt.show()