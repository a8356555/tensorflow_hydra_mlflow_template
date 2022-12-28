import os
import warnings

from omegaconf import DictConfig
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from datetime import datetime
import tensorflow_addons as tfa
from .model_base import ModelBase


class CustomModel(ModelBase):
    def __init__(self, input_shape: tuple, config: DictConfig, num_class: int = 1, enable_accum_grad=False, **kwargs):
        pass