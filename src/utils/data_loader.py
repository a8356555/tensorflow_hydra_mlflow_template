import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from omegaconf import DictConfig
from tensorflow_addons.image import translations_to_projective_transforms, angles_to_projective_transforms, \
    compose_transforms

from .data_type import DataType
from itertools import combinations


def image_normalize(video_data, method: str):
    if method is None:
        print('*' * 5, 'Original image', '*' * 5)
        return video_data

    if method == 'Center':
        print('*' * 5, 'Center', '*' * 5)
        video_data /= 127.5
        video_data -= 1.0
        return video_data

    if method == 'Normalize':
        print('*' * 5, 'Normalize', '*' * 5)
        video_data /= 255.0
        return video_data

    raise ValueError(f'Expected method was Center or Normalize. Received: {method}')


def color_augmentation(data, gamma_adjustment: list):
    assert len(gamma_adjustment) == 2
    print('*' * 5, 'Gamma Color Adjustment was applied!', '*' * 5)
    gamma = tf.random.uniform([], minval=gamma_adjustment[0], maxval=gamma_adjustment[1], dtype=tf.float32)
    data = tf.image.adjust_gamma(data / 255.0, gamma, 255)
    # fill nan value with 0
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
    return data


class TFDataLoader:
    def __init__(self,
                 num_class,
                 cfg: DictConfig,
                 is_train=False,
                 ):
        """
        Load tfrecord dataset
        Args:
            num_class: number of class in dataset
            cfg: config from config.yaml
            is_train:
                - True: training data will add shuffle, data augmentation and combine into batch
                - False: testing data
        """
        self.is_train = is_train
        self.num_class = num_class

        # multi task flag
        self.is_multi_task = cfg.multi_task

        # mode of input frame. [ grayscale, rgb]
        self.color_mode = cfg.color_mode

        # [classification, regression]
        self.task_type = cfg.task_type

        
        self.image_normalize_method = cfg.data_augmentation['image_normalize_method']
        self.translate_percent = cfg.data_augmentation['translate_percent']
        self.random_rotate_degree = cfg.data_augmentation['random_rotate_degree']
        self.gamma_adjustment = cfg.data_augmentation['gamma_adjustment']
        self.horizontal_flip = cfg.data_augmentation['horizontal_flip']
        self.vertical_flip = cfg.data_augmentation['vertical_flip']
        self.resize_size = cfg.data_augmentation['resize_size']
        self.random_crop_size = cfg.data_augmentation['random_crop_size']
        
    def get_data_generator(self, data_path_list, batch_size=1):
        """
        create data generator
        Args:
            data_path_list: numpy file path list
            batch_size: batch size for

        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.TFRecordDataset(data_path_list)
        dataset = dataset.map(self._parse_embryo_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.is_train:
            dataset = dataset.shuffle(buffer_size=128)
            dataset = dataset.batch(batch_size, drop_remainder=False)
        else:
            dataset = dataset.batch(batch_size, drop_remainder=False)
            if self.is_clip_predict:
                dataset = dataset.unbatch()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def _parse_embryo_example(self, example_proto):
        image_feature_description = {
            'data': tf.io.FixedLenFeature([], tf.string),
            'data_type': tf.io.FixedLenFeature([], tf.string),
            'file_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channel': tf.io.FixedLenFeature([], tf.int64),
        }
            
        # Parse the input tf.Example proto using the dictionary above.
        embryo_example = tf.io.parse_single_example(example_proto, image_feature_description)
        
        height = embryo_example['height']
        width = embryo_example['width']
        channels = embryo_example['channel']
        data = tf.io.parse_tensor(embryo_example['data'], 'uint8')
        data = self._raw_processing(data, shape=(height, width, channels))
        
        label = embryo_example['label']
        return data, label

    def _raw_processing(self, data, shape):
        data = tf.cast(data, tf.float32)
        if self.is_train:
            data = self._spatial_augmentation(data, shape)
            if self.gamma_adjustment:
                data = color_augmentation(data, self.gamma_adjustment)
                
        
        data = image_normalize(data, self.image_normalize_method)
        return data


    def _spatial_augmentation(self, data, shape):
        # flip_up_down
        random_num = tf.random.uniform([2], maxval=2, dtype=tf.int32)
        if self.vertical_flip:
            print('*' * 5, 'Random Vertical Flip was applied!', '*' * 5)
            if tf.equal(random_num[0], 0):
                data = tf.map_fn(tf.image.flip_up_down, data)
        # flip_left_right
        if self.horizontal_flip:
            print('*' * 5, 'Random Horizontal Flip was applied!', '*' * 5)
            if tf.equal(random_num[1], 0):
                data = tf.map_fn(tf.image.flip_left_right, data)

        ops = self._affine_transformation_ops(shape)

        if tf.is_tensor(ops):
            data = tfa.image.transform(data, ops, interpolation='BILINEAR')

        return data


    def _affine_transformation_ops(self, shape):
        # translate
        if self.translate_percent:
            print('*' * 5, 'Random Translate was applied!', '*' * 5)
            translate_percent = list(self.translate_percent)
            dx = tf.random.uniform([],
                                   minval=tf.cast(tf.cast(shape[2], tf.float32) * translate_percent[0], tf.int64),
                                   maxval=tf.cast(tf.cast(shape[2], tf.float32) * translate_percent[1], tf.int64),
                                   dtype=tf.int64)
            dy = tf.random.uniform([],
                                   minval=tf.cast(tf.cast(shape[1], tf.float32) * translate_percent[0], tf.int64),
                                   maxval=tf.cast(tf.cast(shape[1], tf.float32) * translate_percent[1], tf.int64),
                                   dtype=tf.int64)
            translate_ops = translations_to_projective_transforms([dx, dy])
        else:
            translate_ops = None

        # rotate
        if self.random_rotate_degree:
            print('*' * 5, 'Random Rotation was applied!', '*' * 5)
            rad = np.array(self.random_rotate_degree) * np.pi / 180.0
            random_num = tf.random.uniform([], minval=rad[0], maxval=rad[1], dtype=tf.float32)
            rotate_ops = angles_to_projective_transforms(random_num, tf.cast(shape[1], tf.float32),
                                                         tf.cast(shape[2], tf.float32))
        else:
            rotate_ops = None

        if tf.is_tensor(translate_ops) and tf.is_tensor(rotate_ops):
            ops = compose_transforms((translate_ops, rotate_ops))
        elif tf.is_tensor(translate_ops):
            ops = translate_ops
        elif tf.is_tensor(rotate_ops):
            ops = rotate_ops
        else:
            ops = None
        return ops