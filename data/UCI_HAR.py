import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
#os.chdir("../..")
#os.chdir('D:/Masters/MachineLearning/Project')

features = list()
with open('UCI HAR Dataset/features.txt') as f:
    features = [line for line in f.readlines()]

train = pd.read_csv('UCI HAR Dataset/train/X_train.txt',sep='\s+', header=None, names=features)

y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', names=['Activity'], squeeze=True)
y_train_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})

#train.dropna()
#train.drop_duplicates()

pca = PCA(n_components=0.99)
print('Original number of features:', train.shape[1])
trainPca = pca.fit_transform(train)

print('Reduced number of features:', trainPca.shape[1])

X_train, X_valid, Y_train, Y_valid = train_test_split(train, y_train, test_size=0.2, random_state=42)
X_trainPca, X_validPca, Y_trainPca, Y_validPca = train_test_split(trainPca, y_train, test_size=0.2, random_state=42)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_valid.npy", X_valid)
np.save("Y_valid.npy", Y_valid)

np.save("X_trainPca.npy", X_trainPca)
np.save("Y_trainPca.npy", Y_trainPca)
np.save("X_validPca.npy", X_validPca)
np.save("Y_validPca.npy", Y_validPca)

test = pd.read_csv('UCI HAR Dataset/test/X_test.txt',sep='\s+', header=None, names=features)
testPca = pca.transform(test)

y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', names=['Activity'], squeeze=True)
y_test_labels = y_train.map({1: 'WALKING', 2:'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING', 5:'STANDING',6:'LAYING'})

#test.dropna()
#test.drop_duplicates()

np.save("X_test.npy", test)
np.save("Y_test.npy", y_test)

np.save("X_testPca.npy", testPca)
np.save("Y_testPca.npy", y_test)

#train['Activity'] = y_train
#train['ActivityName'] = y_train_labels
#test['Activity'] = y_test
#test['ActivityName'] = y_test_labels
#
#plt.title('No of Datapoints per Activity', fontsize=15)
#sns.countplot(train.ActivityName)
#plt.xticks(rotation=90)
#plt.show()

inertialTrainData = []
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt",sep='\s+', header=None).as_matrix())
inertialTrainData.append(pd.read_csv("UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt",sep='\s+', header=None).as_matrix()) 

inertialTrainData = np.transpose(inertialTrainData, (1, 2, 0))

X_trainInert, X_validInert, Y_trainInert, Y_validInert = train_test_split(inertialTrainData, y_train, test_size=0.2, random_state=42)

np.save("inertial_X_train.npy", X_trainInert)
np.save("inertial_Y_train.npy", Y_trainInert)
np.save("inertial_X_valid.npy", X_validInert)
np.save("inertial_Y_valid.npy", Y_validInert)


inertialTestData = []
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/total_acc_x_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/total_acc_y_test.txt",sep='\s+', header=None).as_matrix())
inertialTestData.append(pd.read_csv("UCI HAR Dataset/test/Inertial Signals/total_acc_z_test.txt",sep='\s+', header=None).as_matrix()) 

inertialTestData = np.transpose(inertialTestData, (1, 2, 0))

np.save("inertial_X_test.npy", inertialTestData)
np.save("inertial_Y_test.npy", y_test)
