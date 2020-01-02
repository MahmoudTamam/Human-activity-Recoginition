from keras.applications import resnet, mobilenet_v2
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.models import load_model, Model

class UCI_classifier():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        if self.config.model == 'FC':
            self.model = Sequential()
            self.model.add( Dense(1000, activation='relu') )
            self.model.add( BatchNormalization() )
            self.model.add ( Dropout(0.5) )
            self.model.add( Dense(500, activation='relu') )
            self.model.add( Dense(500, activation='relu') )
            self.model.add( BatchNormalization() )
            self.model.add ( Dropout(0.5) )
            self.model.add( Dense(250, activation='relu') )
            self.model.add( BatchNormalization() )
            self.model.add ( Dropout(0.5) )
            self.model.add( Dense(self.config.classes_num, activation='softmax') )
        else:
            raise NotImplementedError("This Classifier mode is not yet implemented")
        
        input_shape = (None, self.config.input_features)
        self.model.build(input_shape)
        self.model.summary()