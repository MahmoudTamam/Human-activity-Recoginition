from keras.applications import resnet, mobilenet_v2
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Conv1D, Flatten, MaxPooling2D, Dropout, Flatten, BatchNormalization
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
            self.model.add( BatchNormalization() )
            self.model.add ( Dropout(0.5) )
            self.model.add( Dense(250, activation='relu') )
            self.model.add( BatchNormalization() )
            self.model.add ( Dropout(0.5) )
            self.model.add( Dense(self.config.classes_num, activation='softmax') )
            input_shape = (None, self.config.input_features)
        elif self.config.model == 'CNN':
            if self.config.CNN_DIM == '1D':
                self.model = Sequential()
                self.model.add( Conv1D(filters = 18, kernel_size=2, strides=2, padding = 'same', activation='relu') )
                self.model.add( BatchNormalization() )
                self.model.add( Conv1D(filters = 36, kernel_size=2, strides=2, padding = 'same', activation='relu') )
                self.model.add( BatchNormalization() )
                self.model.add( Conv1D(filters = 72, kernel_size=2, strides=2, padding = 'same', activation='relu') )
                self.model.add( BatchNormalization() )
                self.model.add( Conv1D(filters = 144, kernel_size=2, strides=2, padding = 'same', activation='relu') )
                self.model.add( BatchNormalization() )
                self.model.add(Flatten())
                self.model.add( Dense(1000, activation='relu') )
                self.model.add( BatchNormalization() )
                self.model.add ( Dropout(0.5) )
                self.model.add( Dense(500, activation='relu') )
                self.model.add( BatchNormalization() )
                self.model.add ( Dropout(0.5) )
                self.model.add( Dense(250, activation='relu') )
                self.model.add( BatchNormalization() )
                self.model.add ( Dropout(0.5) )
                self.model.add( Dense(self.config.classes_num, activation='softmax') )
                input_shape = (None, self.config.input_features[0], self.config.input_features[1])
            elif self.config.CNN_DIM == '2D':
                raise NotImplementedError("This Classifier mode is not yet implemented")
                #self.model = Sequential( Conv2D() )
        else:
            raise NotImplementedError("This Classifier mode is not yet implemented")
        
        
        self.model.build(input_shape)
        self.model.summary()