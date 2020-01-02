from agents.base import BaseAgent
from graphs.models.UCI_classifier import UCI_classifier
from data_loader.UCI import UCIDataLoader
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras import metrics, optimizers
from keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import os
import keras

class UCI_Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.summary_dir = self.config.summary_dir.replace('/','\\')

        """ Init Callbacks """
        self.callbacks = []
        self.init_callbacks()

        """ Model building: Graph, Loss and optimizer"""
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

        self.model = UCI_classifier(self.config).model
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(lr=self.config.learning_rate),
            metrics=['acc'])
        
        """ init Dataloader """
        self.data_loader = UCIDataLoader(self.config)

        """ initialize training counters """
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_acc:.2f}.hdf5' % self.config.exp_name),
                monitor=self.config.callbacks.checkpoint_monitor, 
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.summary_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )
    
    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        history = self.model.fit(
            x = self.data_loader.X_train,
            y = self.data_loader.Y_train,
            batch_size = self.config.batch_size,
            epochs=self.config.max_epoch,
            verbose=self.config.verbose_training,
            callbacks=self.callbacks,
            validation_split = 0,
            validation_data = (self.data_loader.X_valid, self.data_loader.Y_valid),
            shuffle = True,
            workers = self.config.num_workers,
            use_multiprocessing = True,
            initial_epoch = 0, #TODO:
            validation_freq = 1,
        )
    
    def test(self):
        self.load_checkpoint()

        preds = self.model.predict(
            x = self.data_loader.X_test,
            batch_size= self.config.batch_size,
            verbose = 1
        )
        preds = np.argmax(preds, axis=1)
        cm = confusion_matrix(self.data_loader.Y_test, preds)
        print(cm)
        print(classification_report(self.data_loader.Y_test, preds))

    def load_checkpoint(self):
        self.model.load_weights(self.config.checkpoint_dir+self.config.best_file)

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        pass

    