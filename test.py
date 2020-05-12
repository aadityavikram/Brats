import os
import time
import random
import logging
import numpy as np
from model import unet_9, unet_7
import tensorflow.compat.v1 as tf
from helper import crop_img, color
from plot import plot_original_data, plot_overall, plot_final, save_images, save_slice_images

tf.get_logger().setLevel(logging.ERROR)
tf.disable_v2_behavior()

# global variables
tumor_size = 240
core_size = 64

# directories
data_path = 'data/npz_data/'
train_dir = data_path + '/train/'
val_dir = data_path + '/validation/'
# subject_true = 'BraTS19_TCIA03_216_1'
# subject_false = 'BraTS19_TCIA06_497_1'
subject = random.choice(os.listdir(val_dir))[:-4]
subject_npz_path = val_dir + subject + '.npz'
result_path = 'result/'
subject_plot_path = result_path + subject + '/'


# info of project
def about():
    print('=============================================================')
    print('--> Project = Brain Tumor Segmentation and Detection')
    print('--> Dataset used = MICCAI BraTS 2019 Dataset')
    print('--> Subject currently being observed = {}'.format(subject))
    print('=============================================================')


# performing initial prediction of overall tumor
def initial_prediction(flair, t2):
    model = unet_9(tumor_size)
    model.load_weights('model/full.h5')
    # using Flair and T2 as input for full tumor segmentation
    x = np.zeros(shape=(2, 240, 240), dtype=np.float32)
    x[0], x[1] = flair[90], t2[90]
    x = np.expand_dims(x, axis=0)
    pred_full = model.predict(x)
    resume = True
    if [list(row).count(0) for row in list(pred_full[0][0])].count(pred_full.shape[2]) > 220:
        resume = False
    return pred_full, resume


# performing final prediction of core, ET and full prediction
def final_prediction(t1ce, pred_full):
    # cropping prediction part for tumor core and enhancing tumor segmentation
    crop, li = crop_img(t1ce[90], pred_full[0], 64)
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
def plot_all(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume):
    if not os.path.exists(subject_plot_path):
        os.makedirs(subject_plot_path)
    if resume:
        print('\nTumor Detected\n')
        save_images(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, subject)
        plot_original_data(t1, t2, flair, t1ce, resume, subject_plot_path)
        plot_overall(t2, flair, pred_full, subject_plot_path)
        plot_final(t1, t2, flair, t1ce, pred_full, core, et, tmp, subject_plot_path)
        print('3 plots saved in project_root/{} directory\n'.format(result_path[:-1]))
        print('Name of the 3 plots: -\n1_original_data.png\n2_overall_prediction.png\n3_final_prediction.png')
    else:
        print('\nNo Tumor Detected\n')
        save_images(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, subject)
        plot_original_data(t1, t2, flair, t1ce, resume, subject_plot_path)
        print('Original Data plot saved in project_root/{} directory\n'.format(result_path[:-1]))
        print('Name of the plot: -\n1_original_data.png')


def debug(flair, t1ce, t2):
    print('Loading weights....')
    start_model = time.time()

    model = unet_9(tumor_size)
    model_core = unet_7(core_size)
    model_et = unet_7(core_size)
    model.load_weights('model/full.h5')
    model_core.load_weights('model/core.h5')
    model_et.load_weights('model/et.h5')

    end_model = time.time()
    print('Loaded full, core and et weights in {} s'.format(end_model - start_model))
    print('Performing prediction on all slices....')
    start_global = time.time()
    for slices, (flairs, t2s, t1ces) in enumerate(zip(flair, t2, t1ce)):
        start_slice = time.time()
        x = np.zeros(shape=(2, 240, 240), dtype=np.float32)
        x[0], x[1] = flairs, t2s
        x = np.expand_dims(x, axis=0)
        pred_full = model.predict(x)

        resume = True
        if [list(row).count(0) for row in list(pred_full[0][0])].count(pred_full.shape[2]) > 230:
            resume = False

        # print([list(row).count(0) for row in list(pred_full[0][0])].count(pred_full.shape[2]))
        # print(resume)

        if resume:
            crop, li = crop_img(t1ces, pred_full[0], 64)
            pred_core = model_core.predict(crop)
            pred_et = model_et.predict(crop)
            tmp = color(pred_full[0], pred_core, pred_et, li)
            core, et = tmp.copy(), tmp.copy()
            core[core == 4] = 1
            core[core != 1] = 0
            et[et != 4] = 0

        else:
            core = pred_full[0].copy()
            et = pred_full[0].copy()
            tmp = pred_full[0].copy()

        save_slice_images(pred_full, core, et, tmp, subject, slices)
        end_slice = time.time()
        if slices == 0:
            print('Prediction on 1 slice done in {} s'.format(end_slice - start_slice))
        else:
            print('Prediction on {} slices done in {} s'.format(slices + 1, end_slice - start_slice))
    end_global = time.time()
    print('Performed prediction on all slices in {} s'.format(end_global - start_global))


# wrapper function for entire process
def main():
    core, et, tmp = [], [], []
    subject_npz = np.load(subject_npz_path, allow_pickle=True)['arr_0']
    flair, t1ce, t1, t2 = subject_npz[0], subject_npz[1], subject_npz[2], subject_npz[3]
    pred_full, resume = initial_prediction(flair, t2)
    if resume:
        core, et, tmp = final_prediction(t1ce, pred_full)
    plot_all(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume)
    # debug(flair, t1ce, t2)


# main function
if __name__ == '__main__':
    about()
    main()
