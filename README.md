# Human activity recoginition with Deep learning
Final project for Nile university CIT-651 Machine learning class - Fall 19
Experiments are performed on Pascal VOC 2012, Kaggle statefarm distracted driver and UCI Har dataset.

### UCI HAR Dataset
##### Fully Connected network
Trained on Raw data features 561x1.
![Image description](etc/fig_1.PNG)
![Image description](etc/fig_2.PNG)
##### 1D CNN
Trained with time stamp readings 9x128.
![Image description](etc/fig_3.PNG)

### VOC 2012 Dataset (Indoor, Outdoor, ... )
Trained with 512x512 images with/without bounding box cropping with random augmentation.
![Image description](etc/fig_8.PNG)
![Image description](etc/fig_4.PNG)

### Statefarm dataset (Inside vehicle)
Trained with 480x640 images with random augmentation.
![Image description](etc/fig_5.PNG)
![Image description](etc/fig_6.PNG)

# References:
Template structure and config scripts from [Template link](https://github.com/moemen95/Pytorch-Project-Template)
