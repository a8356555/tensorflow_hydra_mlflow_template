import mlflow
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
from .accum_grad_model import AccumGradModel
from .sam_model import SAMModel, AccumGradSAMModel
from datetime import datetime
from utils.data_type import DataType

class ModelBase:
    model = None
    conv_base = None

    def _warmup(self,
                x: tf.keras.utils.Sequence,
                validation_data: tf.keras.utils.Sequence,
                epochs: int = 1,
                loss: str = 'binary_crossentropy'):
        self.conv_base.trainable = False

        self.model.compile(loss=loss, optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['acc'])
        self.model.fit(x, validation_data=validation_data, epochs=epochs)

        return self.model

    def fit(self, **kwargs):
        loss = kwargs['loss'] if kwargs['loss'] else 'binary_crossentropy'
        if kwargs['warmup'] > 0:
            with mlflow.start_run(nested=True, run_name='warmup'):
                self._warmup(
                    x=kwargs['x'],
                    validation_data=kwargs['validation_data'],
                    loss=loss,
                    epochs=kwargs['warmup'],
                )
        self.conv_base.trainable = kwargs['backbone_net_trainable']
        self.model.compile(loss=loss,
                           optimizer=kwargs['optimizer'],
                           metrics=(kwargs['metrics'] if kwargs['metrics'] else ['acc']))
        with mlflow.start_run(run_id=kwargs['run_id'], nested=True):
            mlflow.tensorflow.autolog(1)
            history = self.model.fit(x=kwargs['x'],
                                     validation_data=kwargs['validation_data'],
                                     batch_size=kwargs['batch_size'],
                                     epochs=kwargs['epochs'],
                                     callbacks=(kwargs['callbacks'] if kwargs['callbacks'] else None))
        return history

    def save(self, filepath, **kwargs):
        self.model.save(filepath, **kwargs)

    def load_model(self, filepath, **kwargs):
        self.model = models.load_model(filepath, **kwargs)
        return self.model

    def evaluate(self, x, **kwargs):
        return self.model.evaluate(x=x, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x=x, **kwargs)

    def __call__(self):
        return self.model

    def get_final_model(self, inputs, outputs, name='I3D', **kwargs):
        if self.enable_accum_grad and self.accum_iters>1:
            model = AccumGradModel(inputs=inputs, outputs=outputs, name=name, **kwargs)
        else:
            # model_class = SAMModel if self.use_sam else models.Model
            model_class = models.Model
            model = model_class(inputs=inputs, outputs=outputs, name=name)
                
        with open('../logs/log.log', 'a') as f:
            print(f'\n\nusing {model_class}')
            f.write(f'[{datetime.now()}] using {model_class}, accum iters: {self.accum_iters}, total batchs: {self.total_batches}\n')
        return model
