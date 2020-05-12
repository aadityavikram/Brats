import os
import shutil
import logging
import numpy as np
import skimage.io as io
from model import unet_9, unet_7
import tensorflow.compat.v1 as tf
from helper import crop_img, color
from plot_app import plot_app_original_data, plot_app_overall, plot_app_final, save_images_app

tf.get_logger().setLevel(logging.ERROR)
tf.disable_v2_behavior()

# global variables
tumor_size = 240
core_size = 64

# directories
result_path = 'result/'


# performing initial prediction of overall tumor
def initial_prediction(flair, t2):
    model = unet_9(tumor_size)
    model.load_weights('model/full.h5')
    # using Flair and T2 as input for full tumor segmentation
    x = np.zeros(shape=(2, 240, 240), dtype=np.float32)
    x[0], x[1] = flair, t2
    x = np.expand_dims(x, axis=0)
    pred_full = model.predict(x)
    resume = True
    if [list(row).count(0) for row in list(pred_full[0][0])].count(pred_full.shape[2]) > 220:
        resume = False
    return pred_full, resume


# performing final prediction of core, ET and full prediction
def final_prediction(t1ce, pred_full):
    # cropping prediction part for tumor core and enhancing tumor segmentation
    crop, li = crop_img(t1ce, pred_full[0], 64)
    model_core = unet_7(core_size)
    model_et = unet_7(core_size)
    model_core.load_weights('model/core.h5')
    model_et.load_weights('model/et.h5')
    pred_core = model_core.predict(crop)
    pred_et = model_et.predict(crop)
    tmp = color(pred_full[0], pred_core, pred_et, li)
    core, et = tmp.copy(), tmp.copy()
    core[core == 4] = 1
    core[core != 1] = 0
    et[et != 4] = 0
    return core, et, tmp


# wrapper plot function
def plot_app_all(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, name):
    if resume:
        # print('\nTumor Detected\n')
        save_images_app(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, name)
        plot_app_original_data(t1, t2, flair, t1ce, resume, name)
        plot_app_overall(t2, flair, pred_full, name)
        plot_app_final(t1, t2, flair, t1ce, pred_full, core, et, tmp, name)
        msg = ['\nTumor Detected\n',
               '3 plots saved in project_root/{} directory\n'.format(result_path[:-1]),
               'Name of the 3 plots: -\n1_original_data.png\n2_overall_prediction.png\n3_final_prediction.png']
        # print(msg[1])
        # print(msg[2])
    else:
        # print('\nNo Tumor Detected\n')
        save_images_app(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, name)
        plot_app_original_data(t1, t2, flair, t1ce, resume, name)
        msg = ['\nNo Tumor Detected\n',
               'Original Data plot saved in project_root/{} directory\n'.format(result_path[:-1]),
               'Name of the plot: -\n1_original_data.png']
        # print(msg[1])
        # print(msg[2])
    return msg


def get_files_from_npz(mri_file_name):
    name = mri_file_name[0][:-4]
    if not os.path.exists('result/{}'.format(name)):
        os.makedirs('result/{}'.format(name))
    mri_file = 'static/data/data/' + mri_file_name[0]
    shutil.copyfile(mri_file, 'result/{}/{}'.format(name, mri_file_name[0]))
    subject_npz = np.load(mri_file, allow_pickle=True)['arr_0']
    flair, t1ce, t1, t2 = subject_npz[0][90], subject_npz[1][90], subject_npz[2][90], subject_npz[3][90]
    return [flair, t1ce, t1, t2, name]


def get_files_from_nii(mri_file_name):
    name = mri_file_name[0].rsplit('_', 1)[0]
    if not os.path.exists('result/{}'.format(name)):
        os.makedirs('result/{}'.format(name))
    files_list = []
    for item in mri_file_name:
        mri_file = 'static/data/data/' + item
        img = io.imread(mri_file, plugin='simpleitk')
        img = (img - img.mean()) / img.std()
        img = img.astype('float32')
        img = img[90]
        img = img.reshape((1,) + img.shape)
        files_list.append(img)
        shutil.copyfile(mri_file, 'result/{}/{}'.format(name, item))
    flair, t1ce, t1, t2 = files_list[0], files_list[1], files_list[2], files_list[3]
    return [flair, t1ce, t1, t2, name]


def main(mri_file_name):
    core, et, tmp = [], [], []
    files_list = get_files_from_npz(mri_file_name) if len(mri_file_name) == 1 else get_files_from_nii(mri_file_name)
    flair, t1ce, t1, t2, name = files_list[0], files_list[1], files_list[2], files_list[3], files_list[4]
    pred_full, resume = initial_prediction(flair, t2)
    if resume:
        core, et, tmp = final_prediction(t1ce, pred_full)
    msg = plot_app_all(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, name)
    return msg
