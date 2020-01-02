from agents.base import BaseAgent
from graphs.models.classifier import Actvity_classifier
from data_loader.voc import VOCDataLoader
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras import metrics, optimizers
from keras.optimizers import SGD, Adam
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import tensorflow as tf
import numpy as np
import os
import keras

class VOCACTIVITY_Agent(BaseAgent):
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

        self.model = Actvity_classifier(self.config).model
        
        if self.config.single_label == True:
            self.model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=Adam(lr=self.config.learning_rate),
                metrics=['accuracy'])
        else:
            self.model.compile(
                loss='binary_crossentropy',
                optimizer=Adam(lr=self.config.learning_rate),
                metrics=['accuracy'])

        """ init Dataloader """
        self.data_loader = VOCDataLoader(self.config)

        self.train_generator = self.data_loader.train_generator()
        self.valid_generator = self.data_loader.valid_generator()

        self.train_iters = self.data_loader.train_iters
        self.valid_iters = self.data_loader.valid_iters

        """ initialize training counters """
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor=self.config.callbacks.checkpoint_monitor, #TODO: Change into accuracy
                mode=self.config.callbacks.checkpoint_mode, #TODO: Change into  max with accuracy
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
        if self.config.use_scheduler == True:
            raise NotImplementedError ("Scheduler not yet implemented")
            self.callbacks.append(
                LearningRateScheduler(schedule = self.config.scheduler.step_decay, verbose=0)
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
        history = self.model.fit_generator(
            generator = self.train_generator,
            steps_per_epoch = self.train_iters,
            epochs=self.config.max_epoch,
            verbose=self.config.verbose_training,
            callbacks=self.callbacks,
            validation_data = self.valid_generator,
            validation_steps = self.valid_iters,
            validation_freq = 1,
            workers = self.config.num_workers,
            use_multiprocessing = False,
            shuffle = False,
            initial_epoch = 0#TODO
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])

    def test(self):
        pass
    
    def load_checkpoint(self):
        self.model.load_weights(self.config.checkpoint_dir+self.config.best_file)

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        pass

    