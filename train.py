import os
import logging
import numpy as np
from helper import crop_img
import matplotlib.pyplot as plt
from model import unet_9, unet_7
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel(logging.ERROR)

tf.disable_v2_behavior()

# global variables
tumor_size = 240
core_size = 64

# paths
data_path = 'data/npz_data'


def create_data():
    flair = np.load(os.path.join(data_path, 'train_flair.npz'), allow_pickle=True)['arr_0']
    t1ce = np.load(os.path.join(data_path, 'train_t1ce.npz'), allow_pickle=True)['arr_0']
    t1 = np.load(os.path.join(data_path, 'train_t1.npz'), allow_pickle=True)['arr_0']
    t2 = np.load(os.path.join(data_path, 'train_t2.npz'), allow_pickle=True)['arr_0']
    seg = np.load(os.path.join(data_path, 'train_seg.npz'), allow_pickle=True)['arr_0']
    return flair, t1ce, t1, t2, seg


def plot(results, name):
    plt.figure(figsize=(8, 8))
    plt.plot(results.history["loss"], label="training loss")
    plt.plot(results.history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('result/', name + '.png')
    plt.show()


def train_overall(flair, t2, seg):
    x = np.zeros((2, 240, 240), np.float32)
    x[0], x[1] = flair, t2
    x = np.expand_dims(x, axis=0)

    y = np.zeros((1, 240, 240), np.float32)
    y[0] = seg
    y = np.expand_dims(y, axis=0)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1)
    model = unet_9(tumor_size)
    results = model.fit(x_train, y_train, batch_size=32, epochs=300, validation_data=(x_valid, y_valid))
    pred_full = model.predict(x)
    model.save_weights('model/full.h5')
    plot(results, 'plot_overall')
    return pred_full


def train_final(t1ce, pred_full, seg):
    crop, li = crop_img(t1ce, pred_full, 64)
    x = np.zeros((1, 240, 240), np.float32)
    x[0] = crop
    x = np.expand_dims(x, axis=0)

    y = np.zeros((1, 240, 240), np.float32)
    y[0] = seg
    y = np.expand_dims(y, axis=0)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1)
    model = unet_7(core_size)
    results = model.fit(x_train, y_train, batch_size=32, epochs=300, validation_data=(x_valid, y_valid))
    pred_core = model.predict(crop)
    pred_et = model.predict(crop)
    model.save_weights('model/core.h5')
    model.save_weights('model/et.h5')
    plot(results, 'plot_final')
    return pred_core, pred_et


def main():
    flair, t1ce, t1, t2, seg = create_data()
    pred_full = train_overall(flair, t2, seg)
    pred_core, core_et = train_final(t1ce, pred_full, seg)
