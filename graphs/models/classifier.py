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
            model = resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(self.config.resize[0], self.config.resize[1], 3), pooling='avg', classes=1000)
            x = model.output
            x = Dense(512, activation='relu')(x) #add new layer
            x = Dropout(0.5)(x) #add new layer
            x = Dense(512, activation='relu')(x) #add new layer
            x = Dropout(0.5)(x) #add new layer
            if self.config.single_label == True:
                out = Dense(self.config.classes_num, activation='softmax', name='output_layer')(x)
            else:
                out = Dense(self.config.classes_num, activation='sigmoid', name='output_layer')(x)
            self.model = Model(inputs=model.input,outputs= out)
        else:
            raise NotImplementedError("This Classifier mode is not yet implemented")

        self.model.summary()