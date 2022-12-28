from importlib import import_module

from tensorflow.keras import optimizers
from tensorflow.keras.losses import Reduction
from tensorflow_addons.losses import SigmoidFocalCrossEntropy


def import_model(config: dict):
    module_path = '.'.join(('models', config['model_file_name']))
    model_class = config['model_class_name']
    mod = import_module(module_path, package=model_class)
    mod = getattr(mod, model_class)
    return mod


def import_optimizers(config: dict):
    optimizer_name = config['optimizer_name']
    opt = getattr(optimizers, optimizer_name)
    # remove optimizer_name
    config.pop('optimizer_name')
    opt = opt(**config)
    print('Optimizer:', opt.get_config())
    return opt

def import_loss(loss_name: str):
    if loss_name == 'sigmoid_focal_crossentropy':
        return SigmoidFocalCrossEntropy(reduction=Reduction.AUTO)
    return loss_name
