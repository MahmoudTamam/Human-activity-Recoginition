from agents.base import BaseAgent
from graphs.models.classifier import Actvity_classifier
from data_loader.voc import VOCDataLoader
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras import metrics, optimizers
from keras.optimizers import SGD, Adam
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, classification_report
import numpy as np
import os
import keras

class VOCACTIVITY_Agent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #self.config.summary_dir = self.config.summary_dir.replace('/','\\')

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
                filepath=os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_acc:.2f}.h5' % self.config.exp_name),
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
        self.load_checkpoint()
        
        print (self.model.evaluate_generator(
            generator = self.valid_generator,
            steps = self.valid_iters,
            verbose = 1,
            use_multiprocessing = False,
            workers = self.config.num_workers
        ))
        
        self.valid_generator = self.data_loader.valid_generator()
        preds = self.model.predict_generator(
            generator = self.valid_generator,
            steps = self.valid_iters,
            verbose = 1,
            workers=self.config.num_workers, 
            use_multiprocessing=False
        )
        
        if self.config.single_label == True:
            preds = np.argmax(preds, axis=1)
            cm = confusion_matrix(self.data_loader.valid_true, preds)
        else:
            preds = np.where(preds >= 0.5, 1, 0)
            cm = multilabel_confusion_matrix(self.data_loader.valid_true, preds)
        
        print(cm)
        print(classification_report(self.data_loader.valid_true, preds))
        acc_samples = accuracy_score(self.data_loader.valid_true, preds, normalize=False)
        acc_score = accuracy_score(self.data_loader.valid_true, preds, normalize=True)
        print(acc_samples)
        print(acc_score)

    def load_checkpoint(self):
        self.model = load_model(self.config.checkpoint_dir+self.config.best_file)

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        pass

    