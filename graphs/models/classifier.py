from keras.applications import resnet
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class Actvity_classifier():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        if self.config.model == 'ResNet50':
            self.model = resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling='avg', classes=1000)
            model_config = self.model.get_config()
            model_config['layers'][0]['config']['batch_input_shape'] = (None, 500, 500, 3)
            model_config['layers'][-1]['config']['units'] = self.config.classes_num
            model_config['layers'][-1]['config']['activation'] = 'sigmoid'
            self.model = self.model.from_config(model_config)
        else:
            raise NotImplementedError("This Classifier mode is not yet implemented")

        self.model.summary()