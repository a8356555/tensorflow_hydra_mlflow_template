from cgi import test
import os
from datetime import datetime
from pathlib import Path
from pyexpat import model

import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow.keras.backend as K
from omegaconf import DictConfig
from sklearn import metrics
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import AUC
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow_addons.metrics import F1Score

from utils.callbacks import HalfPeriodCosineSchedule, OneCycleScheduler
# from utils.cmatrix import save_confusion_matrix
from utils.data_loader import TFDataLoader
from utils.data_type import DataType
from utils.dynamic_import import import_model, import_optimizers, import_loss
from models.accum_grad_model import AccumGradModel
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

custom_model_objects = {'AccumGradModel': AccumGradModel}

def load_callbacks(weights_folder_path: str, ckpt_description: str, run_id: str, learning_rate, config: DictConfig,
                   **kwargs) -> list:
    callbacks_list = []
    if config.task_type == 'classification':
        checkpointer_auc = callbacks.ModelCheckpoint(filepath=os.path.join(weights_folder_path,
                                                                           ckpt_description + '_auc.h5'),
                                                     save_best_only=True,
                                                     monitor='val_main_prediction_auc' if config.multi_task else 'val_auc',
                                                     mode='max')
        checkpointer_f1 = callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_folder_path, ckpt_description + '_f1.h5'),
            save_best_only=True,
            monitor='val_main_prediction_f1_score' if config.multi_task else 'val_f1_score',
            mode='max')

        callbacks_list.append(checkpointer_auc)
        callbacks_list.append(checkpointer_f1)

    else:
        checkpointer_regression = callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_folder_path, ckpt_description + '_regression.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min')
        callbacks_list.append(checkpointer_regression)

    tensorboard_callback = callbacks.TensorBoard(log_dir=hydra.utils.to_absolute_path('./logs/tfd/{}'.format(run_id)),
                                                 profile_batch=0,
                                                 write_images=True,
                                                 histogram_freq=1)

    callbacks_list.append(tensorboard_callback)
    if 'RLROP' in config and config.RLROP.enable:    
        print('*' * 10 + 'Using RLROP' + '*' * 10)
        
        rlrop = callbacks.ReduceLROnPlateau(
            monitor=config.RLROP.monitor, factor=config.RLROP.factor, patience=config.RLROP.patience,
            mode=config.RLROP.mode, min_delta=config.RLROP.min_delta, cooldown=config.RLROP.cooldown
        )
        callbacks_list.append(rlrop)
    
    if 'SGDWR' in config and config.SGDWR.enable:
        print('*' * 10 + 'Using SGDWR' + '*' * 10)

        def sgdwr_schedule(epoch):
            # decay within cycle
            lr = learning_rate * (config.SGDWR.decay**(epoch % config.SGDWR.period))
            lr = max(lr, 1e-5)  # avoid too small lr
            return lr

        sgdwr = callbacks.LearningRateScheduler(sgdwr_schedule)
        callbacks_list.append(sgdwr)

    if 'OCSGDWR' in config and config.OCSGDWR.enable:
        def modified_sgdwr_schedule(epoch):
            if epoch < config.OCSGDWR.first_cycle:
                lr = learning_rate * (1-epoch*0.5/config.OCSGDWR.first_cycle)
            else:
                lr = 0.5*learning_rate*(config.OCSGDWR.decay**(epoch-config.OCSGDWR.first_cycle % config.n_epochs))
            lr = max(lr, 1e-5)  # avoid too small lr
            return lr

        modified_sgdwr = callbacks.LearningRateScheduler(modified_sgdwr_schedule)
        callbacks_list.append(modified_sgdwr)
    
    
    if 'HPCS' in config and config.HPCS.enable:
        print('*' * 10 + 'Using Half-period Cosine Schedule' + '*' * 10)
        hpcs = HalfPeriodCosineSchedule(
            min_lr=config.HPCS.initial_lr,
            max_lr=learning_rate,
            filepath=os.path.join(weights_folder_path, ckpt_description + '_hpcs.h5'),
            lr_decay=config.HPCS.decay,
            cycle_length=config.HPCS.period,
            multi_factor=config.HPCS.multi_factor,
            warm_up_epoch=config.HPCS.warmup,
        )
        callbacks_list.append(hpcs)

    if 'OCS' in config and config.OCS.enable:
        print('*' * 10 + 'Using One Cycle Schedule' + '*' * 10)
        ocs = OneCycleScheduler(
            max_lr=learning_rate,
            steps=np.ceil(kwargs['total_batches'] * config.n_epochs),
            div_factor=config.OCS.div_factor
        )
        callbacks_list.append(ocs)    
    
    
    
    
    return callbacks_list


def classification_evaluation(model_object, test_datagen, num_class, ckpt_description, project_path, cfg: DictConfig, seed=666):
    print('*' * 10 + 'Evaluation' + '*' * 10)
    predictions = model_object.predict(test_datagen)
    if cfg.multi_task:
        predictions = np.array(predictions)[0, :]
    predictions = np.reshape(predictions, (-1, num_class))
    print('predictions', predictions.shape)
    ground_truth = []
    for element in test_datagen.as_numpy_iterator():
        if cfg.multi_task:
            ground_truth.append(element[1]['main_prediction'])
        else:
            ground_truth.append(element[1])
    ground_truth = np.squeeze(np.array(ground_truth))
    ground_truth = np.reshape(ground_truth, (-1, num_class))
    print('ground_truth', ground_truth.shape)
    if num_class == 1:
        best_predictions = predictions > 0.5
        best_true_labels = ground_truth > 0.5
        f1_flag = 'binary'
        num_class += 1
    else:
        best_predictions = np.argmax(predictions, axis=-1).reshape(-1, 1)
        best_true_labels = np.argmax(ground_truth, axis=-1)
        f1_flag = 'macro'

    print('Test accuracy: {:.4f}'.format(metrics.accuracy_score(best_true_labels, best_predictions)))
    print('Test f1 score: {:.4f}'.format(metrics.f1_score(best_true_labels, best_predictions, average=f1_flag)))
    print('Test AUC: {:.4f}'.format(metrics.roc_auc_score(ground_truth, predictions, average='macro',
                                                          multi_class='ovr')))
    cmatrix = metrics.confusion_matrix(best_true_labels, best_predictions, labels=np.arange(num_class))
    print(cmatrix)

    # save_confusion_matrix(cmatrix, list(np.arange(num_class)), ckpt_description, project_path=project_path)
    import pandas as pd
    pd.DataFrame({'pred': list(predictions), 'gt': list(ground_truth)}).to_csv(os.path.join(project_path, 'reports/outputs', f'{seed}_{ckpt_description.split("_")[-1]}.csv'))


def regression_evaluation(model_object, test_datagen, ckpt_description):
    print('*' * 10 + 'Evaluation' + '*' * 10)
    predictions = model_object.predict(test_datagen)
    print(predictions.shape)
    predictions = predictions.flatten()
    print('predictions', predictions.shape)
    ground_truth = []
    for element in test_datagen.as_numpy_iterator():
        ground_truth.append(element[1])
    ground_truth = np.squeeze(np.array(ground_truth))
    print('ground truth', ground_truth.shape)

    plt.scatter(ground_truth, predictions)
    plt.axline(xy1=(0, 0), slope=1, color='black', ls='--', lw=1, alpha=.5)
    plt.title(ckpt_description)
    plt.savefig(hydra.utils.to_absolute_path(f'reports/figures/{ckpt_description}.png'))

    print(f'MAE: {metrics.mean_absolute_error(ground_truth, predictions)}')
    print(f'MSE: {metrics.mean_squared_error(ground_truth, predictions)}')

def make_all_dir(project_path):
    todo_paths = [
        os.path.join(project_path, 'reports/outputs'),
        os.path.join(project_path, 'reports/input_check'),
                 ]
    for path in todo_paths:
        if not os.path.exists(path):
            os.mkdir(path)

@hydra.main(config_path='configs', config_name='config', strict=False)
def main(cfg: DictConfig) -> None:
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)
    # set mlflow log uri
    mlflow.set_tracking_uri(cfg.mlflow.url)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # train config
    batch_size = cfg.batch_size
    assert isinstance(batch_size, int)
    cfg.pop('batch_size')
    n_epochs = cfg.n_epochs
    train_warmup = cfg.train_strategy.warmup
    backbone_net_trainable = cfg.train_strategy.backbone_net_trainable
    enable_accum_grad = cfg.train_strategy.enable_accum_grad
    
    
    num_class = cfg.data.num_class

    # mixed precision
    if cfg.mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        
    # get project path (as hydra change cwd)
    project_path = hydra.utils.get_original_cwd()
    
    make_all_dir(project_path)
    ckpt_description = ('Model_{}_{}'.format(cfg.model.model_class_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    weights_folder_path = os.path.join(project_path, 'model_weights')
    seed = cfg.data.dataset_name.split('_seed')[-1]
    # get dataset in tfrecords format
    
    dataset_path = Path(
        hydra.utils.to_absolute_path(os.path.join('datasets/final/tensorflow', cfg.data.dataset_name))) 
   
    dataset_path_dict = {
        'train': list(map(lambda x: x.as_posix(), sorted(dataset_path.glob('train_*.tfrecords')))),
        'val': list(map(lambda x: x.as_posix(), sorted(dataset_path.glob('val_*.tfrecords')))),
        'test': list(map(lambda x: x.as_posix(), sorted(dataset_path.glob('test_*.tfrecords'))))
    }

    # load dataset
    train_datagen = TFDataLoader(num_class=num_class,
                                 cfg=cfg,
                                 is_train=True).get_data_generator(dataset_path_dict['train'] +
                                                                   dataset_path_dict['val'],
                                                                   batch_size=batch_size)
    test_datagen = TFDataLoader(num_class=num_class,
                                cfg=cfg,
                                is_train=False).get_data_generator(dataset_path_dict['test'])
    
    if not os.path.exists(os.path.join(project_path, f'reports/input_check')):
        os.mkdir(os.path.join(project_path, f'reports/input_check'))

    total_batches = 0
    for xs, ys in train_datagen:
        total_batches += 1
        if total_batches == 1:
            print(ys)
            x = xs[0]
            tf_images = ((x.numpy()+1)*127.5).astype(np.uint8)
            np.save(os.path.join(project_path, f'reports/input_check/{datetime.now()}.npy'), tf_images)

        if total_batches % 10 == 0:
            print('counting train batches', total_batches)

    # get video resolution
    one_example_dataset = train_datagen.take(1)
    shape = one_example_dataset.as_numpy_iterator().next()[0].shape
    print('Train data resolution:', one_example_dataset.as_numpy_iterator().next()[0].shape)
    one_example_dataset = test_datagen.take(1)
    print('Test data resolution:', one_example_dataset.as_numpy_iterator().next()[0].shape)

    # dynamically import model class
    model_class = import_model(cfg.model)
    model_kwargs = {
        'accum_iters': cfg.train_strategy.global_batch_size//batch_size,
        'total_batches': total_batches
    }
    model_object = model_class(
        input_shape=shape,
        num_class=num_class,
        config=cfg,
        project_path=project_path,
        enable_accum_grad=enable_accum_grad,
        **model_kwargs
    )
    # This is what our model looks like now:
    model_object.model.summary(line_length=120)

    # setting optimizer
    optimizer = import_optimizers(config=cfg.optimizer)
    optimizer_learning_rate = cfg.optimizer.learning_rate
    
    #####################
    # start training    #
    #####################
    with mlflow.start_run() as run:
        K.clear_session()
        print('run_id:', run.info.run_id)
        with open(os.path.join(project_path, 'logs/log.log'), 'a') as f:
            f.write(f'[{datetime.now()}] run_id {run.info.run_id} \n')
        # log experiment tag based trello
        ckpt_description = ckpt_description + '_' + run.info.run_id
        if cfg.mlflow.tag is not None:
            print('exp: ', cfg.mlflow.tag)
            mlflow.set_tag('exp', cfg.mlflow.tag)
        cfg.pop('mlflow')
        
        callback_kwargs = {
            'total_batches': total_batches,
            'batch_size': batch_size
        }

        callback_list = load_callbacks(
            weights_folder_path,
            ckpt_description,
            run_id=run.info.run_id,
            learning_rate=optimizer_learning_rate,
            config=cfg,
            **callback_kwargs
        )
        if cfg.task_type == 'classification':
            if num_class > 1:
                # multi-class
                auc_metric = AUC(multi_label=True, num_thresholds=10000)
                f1_metric = F1Score(num_class, average='macro')
            else:
                # binary
                auc_metric = AUC(multi_label=False, num_thresholds=10000)
                f1_metric = F1Score(num_class, average='macro', threshold=0.5)
            metric_list = ['acc', auc_metric, f1_metric]
            # TODO
            if cfg.multi_task:
                metric_dict = {'main_prediction': metric_list}
                for task in cfg.model.aux_task_names:
                    if task in ['age', 'te', 'expansion']:
                        metric_dict[f'{task}_prediction'] = ['mae', 'mse']
                    else:
                        metric_dict[f'{task}_prediction'] = metric_list
        else:
            metric_list = ['mae', 'mse']
        # remove redundancy params
        cfg.pop('optimizer')
        # avoid params too long to store into mlflow
        mlflow.log_params(cfg.pop('model'))
        mlflow.log_params(cfg.pop('data'))
        mlflow.log_params(cfg.pop('train_strategy'))
        mlflow.log_params(cfg.pop('data_augmentation'))
        # log remaining params
        mlflow.log_params(cfg)

        mlflow.tensorflow.autolog(1)
        model_object.fit(
            x=train_datagen,
            validation_data=test_datagen,
            batch_size=batch_size,
            loss=import_loss(cfg.loss),
            optimizer=optimizer,
            epochs=n_epochs,
            callbacks=callback_list,
            metrics=metric_dict if cfg.multi_task else metric_list,
            run_id=run.info.run_id,
            warmup=train_warmup,
            backbone_net_trainable=backbone_net_trainable,
        )
        print('*' * 10 + 'Training Finished' + '*' * 10)

    # save model
    model_path = os.path.join(weights_folder_path, ckpt_description + '_wockpt.h5')
    model_object.save(model_path)
    print('*' * 10 + 'Save model' + '*' * 10)
    # load model
    model_object.load_model(model_path, custom_objects=custom_model_objects)
    print('*' * 10 + 'Load model' + '*' * 10)

    #################
    # evaluation    #
    #################
    if cfg.task_type == 'classification':
        classification_evaluation(model_object=model_object, test_datagen=test_datagen, num_class=num_class,
                                  ckpt_description=ckpt_description, project_path=project_path, cfg=cfg, seed=seed)
    if cfg.task_type == 'regression':
        regression_evaluation(model_object=model_object, test_datagen=test_datagen,
                              ckpt_description=ckpt_description)


if __name__ == '__main__':
    main()
