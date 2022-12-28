import warnings

import numpy as np
from sklearn import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt


class ModelCheckpoint_auc_or_f1(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """
    def __init__(self,
                 filepath,
                 val_datagen,
                 monitor='auc',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 period=1):
        super(ModelCheckpoint_auc_or_f1, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.val_datagen = val_datagen
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                # change here: calculate metric you want here at the end of epoch and compare it with previous results
                preds = self.model.predict_generator(self.val_datagen)
                preds_class = (preds > .5) * 1
                labels = self.val_datagen.y
                if self.monitor == 'auc':
                    current = metrics.roc_auc_score(labels, preds)
                elif self.monitor == 'f1':
                    current = metrics.f1_score(labels, preds_class)
                print('Validation {0} of this epoch: {1:8.4f}'.format(self.monitor, current))

                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class HalfPeriodCosineSchedule(Callback):
    def __init__(self, min_lr, max_lr, filepath, lr_decay=0.8, cycle_length=10, multi_factor=2.0, warm_up_epoch=5):

        super(HalfPeriodCosineSchedule, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.cycle_length = cycle_length
        self.multi_factor = multi_factor
        self.warm_up_epoch = warm_up_epoch
        self.filepath = filepath
        self.previous_restart_epoch = 0

        if warm_up_epoch == 0:
            self.is_warming = False
            self.restart_epoch = cycle_length
        else:
            self.is_warming = True
            self.restart_epoch = warm_up_epoch + cycle_length

    def sgdr_lr(self, epoch):
        # lr_{min}+0.5*(lr_{max}-lr_{min})*(1+cos(\frac{(epochs-warm\_up\_epoch)}{cycle\_length}*\pi)
        fraction_to_restart = (epoch - self.previous_restart_epoch) / (self.cycle_length - 1)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def warm_lr(self, epoch):
        # lr_{min}+(lr_{max}-lr_{min})*(\frac{epochs}{warm\_up\_epochs})
        lr = self.min_lr + (self.max_lr - self.min_lr) * (epoch / self.warm_up_epoch)
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if self.is_warming:
            K.set_value(self.model.optimizer.learning_rate, self.warm_lr(epoch))
        else:
            K.set_value(self.model.optimizer.learning_rate, self.sgdr_lr(epoch))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.learning_rate)
        if self.is_warming:
            if epoch + 1 == self.warm_up_epoch:
                # warm up end
                self.is_warming = False
                self.previous_restart_epoch = self.warm_up_epoch
        else:
            if epoch + 1 == self.restart_epoch:
                # reset state
                self.cycle_length = np.ceil(self.cycle_length * self.multi_factor)
                self.previous_restart_epoch = self.restart_epoch
                self.restart_epoch += self.cycle_length
                self.max_lr *= self.lr_decay
                self.model.save(self.filepath.format(epoch=epoch))



class CosineAnnealer:
    
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    def __init__(self, max_lr, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = max_lr / div_factor
        final_lr = max_lr / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps
        
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        
        self.phases = [[CosineAnnealer(lr_min, max_lr, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                 [CosineAnnealer(max_lr, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
            
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())
        
    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        except AttributeError:
            return None
        
    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None
        
    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        except AttributeError:
            pass # ignore
        
    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]
    
    def mom_schedule(self):
        return self.phases[self.phase][1]
    
    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title('Learning Rate')
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title('Momentum')
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.learning_rate)