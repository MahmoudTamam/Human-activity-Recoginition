from keras.applications import resnet, mobilenet_v2
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import load_model, Model

class Actvity_classifier():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        if self.config.model == 'ResNet50':
            pass
        else:
            raise NotImplementedError("This Classifier mode is not yet implemented")

        self.model.summary()