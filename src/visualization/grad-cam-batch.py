"""
this Grad-CAM code is referenced from keras example in following
https://keras.io/examples/vision/grad_cam/
"""

import os
from pathlib import Path

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(model_path: Path, data_path: str, info_file_path, last_conv_layer_name, label_column, random_state) -> None:
    if info_file_path.find('xlsx') != -1:
        df = pd.read_excel(info_file_path)
    elif info_file_path.find('csv') != -1:
        df = pd.read_excel(info_file_path)
    else:
        raise Exception('info file type is not supported!')

    df = df[['id', label_column]]
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)

    # get dataset in tfrecords format
    data = np.load(os.path.join(data_path, f"raw_{train_df.iloc[0]['id']}.npy"))
    # add channel dimension
    data = np.expand_dims(data, axis=-1)
    # add batch dimension
    data = np.expand_dims(data, axis=0)
    _, t, h, w, c = data.shape

    # load model
    model = load_model(model_path, custom_objects={'F1Score': F1Score})
    if model.input.shape[-1] != 1:
        data = np.broadcast_to(data, shape=(1, t, h, w, model.input.shape[-1]))
    for layer in model.layers:
        print(layer.name)

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    layer_name_list = [layer.name for layer in model.layers]
    classifier_layer_names = layer_name_list[min(layer_name_list.index(last_conv_layer_name) +
                                                 1, len(layer_name_list)):]
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    heatmap, prediction = make_gradcam_heatmap(data, last_conv_layer_model, classifier_model)
    heatmap = np.uint8(255 * heatmap)
    # use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")
    # use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    map_t, map_h, map_w, map_c = jet_heatmap.shape

    # create an image with RGB colorized heatmap
    output = []
    for i, img in enumerate(data[0]):
        coefficient = i / t * map_t - int(i / t * map_t)
        if coefficient == 0:
            heat_frame = jet_heatmap[int(i / t * map_t)]
        else:
            heat_frame = (jet_heatmap[int(i / t * map_t)] * coefficient + jet_heatmap[int(i / t * map_t)] *
                          (1 - coefficient))

        heat_frame = keras.preprocessing.image.array_to_img(heat_frame)
        heat_frame = heat_frame.resize((h, w))
        heat_frame = keras.preprocessing.image.img_to_array(heat_frame)
        superimposed_img = heat_frame * 0.5 + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        output.append(superimposed_img)

    fig, axs = plt.subplots(int(np.ceil(t / 10)), 10, dpi=300)
    for i in range(t):
        axs[i // 10][i % 10].matshow(output[i])

    for axs_list in axs:
        for ax in axs_list:
            ax.set_axis_off()
    label = df[label_column].values
    fig.suptitle('{} \nGT: {}\nPrediction: {}'.format(Path(data_path).name, label, prediction[0]), fontsize=6)
    plt.savefig('reports/grad-cam/{}.png'.format(output_name))

    # Output video
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('{}.avi'.format(output_name), fourcc, 24, (w, h))
    for frame in output:
        out.write(np.uint8(cv2.cvtColor(np.float32(frame), cv2.COLOR_RGB2BGR)))
    # Release everything if job is finished
    out.release()


def make_gradcam_heatmap(img_array, last_conv_layer_model: keras.models.Model, classifier_model):
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        img_array = img_array / 255.0
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    print(pooled_grads)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, preds


if __name__ == '__main__':
    info_file_path = "datasets/raw/lwh/Book2_revise_v2-4.xlsx"
    model_path = Path("model_weights/Model_I3DInception_2021-08-18_16-51-26_3d78a102dc534f77919ae4fe54c76d5c_wockpt.h5")
    data_path = "datasets/interim/w100h100_t209_0055_1100"
    last_conv_layer_name = "Mixed_5c"
    output_name = "output3"
    label_column = "diploid-aneuploid grouping binary"
    random_state = 42

    main(info_file_path=info_file_path, model_path=model_path, data_path=data_path,
         last_conv_layer_name=last_conv_layer_name, label_column=label_column, random_state=random_state)
