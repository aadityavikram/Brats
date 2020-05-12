import os
import time
import psutil
import imageio
import numpy as np
import skimage.io as io
from model import unet_9, unet_7
from helper import crop_img, color

# declaring paths
data_path = 'data/'
original_data_path = os.path.join(data_path, 'original_data')
train_dir = original_data_path + '/MICCAI_BraTS_2019_Data_Training/'
val_dir = original_data_path + '/MICCAI_BraTS_2019_Data_Validation/'
npz_path = data_path + '/npz_data/'


def create_names_list():
    train_names_path = os.path.join(original_data_path, 'MICCAI_BraTS_2019_Data_Training\All')
    val_names_path = os.path.join(original_data_path, 'MICCAI_BraTS_2019_Data_Validation')

    train_names, val_names = [], []
    for files in os.listdir(train_names_path):
        train_names.append(files)
    for files in os.listdir(val_names_path):
        if files == 'name_mapping_validation_data.csv' or files == 'survival_evaluation.csv':
            continue
        val_names.append(files)
    np.save(os.path.join(data_path, 'train_names.npy'), train_names)
    np.save(os.path.join(data_path, 'val_names.npy'), val_names)


# create flair, t1ce, t1 and t2 data of a subject
def create_data(src):
    n_slices = 155
    for file in src:
        imgs = []
        img = io.imread(file, plugin='simpleitk')
        img = (img - img.mean()) / img.std()
        img = img.astype('float32')
        for slices in range(n_slices):
            img_t = img[slices]
            img_t = img_t.reshape((1,) + img_t.shape)
            img_t = img_t.reshape((1,) + img_t.shape)
            for n in range(img_t.shape[0]):
                imgs.append(img_t[n])
    return np.array(imgs)


def save_npz_one_subject(subject, mode, idx):
    combined = []
    if mode == 'train':
        subject_dir = train_dir + 'All/' + subject + '/' + subject
    else:
        subject_dir = val_dir + '/' + subject + '/' + subject
    combined.append(create_data(subject_dir + '_flair.nii.gz'))
    combined.append(create_data(subject_dir + '_t1ce.nii.gz'))
    combined.append(create_data(subject_dir + '_t1.nii.gz'))
    combined.append(create_data(subject_dir + '_t2.nii.gz'))
    if mode == 'train':
        subject_path = npz_path + '/train/'
    else:
        subject_path = npz_path + '/validation/'
    if not os.path.exists(subject_path):
        os.makedirs(subject_path)
    if mode == 'train':
        combined.append(create_data(subject_dir + '_seg.nii.gz'))
    np.savez_compressed(os.path.join(subject_path, subject + '.npz'), combined)
    if idx % 20 == 0 and idx != 0:
        print('Saved {} {} data'.format(idx, mode))


def save_all_npz():
    train_sub_list = np.load(os.path.join(data_path, 'train_names.npy'))
    val_sub_list = np.load(os.path.join(data_path, 'val_names.npy'))
    start_global = time.time()
    start = time.time()
    print('Saving train data')
    for i, sub in enumerate(train_sub_list):
        save_npz_one_subject(sub, 'train', i)
    end = time.time()
    print('Saved training data | Time elapsed = {} s\n'.format(end - start))
    print('Saving validation data')
    start = time.time()
    for i, sub in enumerate(val_sub_list):
        save_npz_one_subject(sub, 'validation', i)
    end = time.time()
    print('Saved validation data | Time elapsed = {} s\n'.format(end - start))
    end_global = time.time()
    print('Saved train and validation data | Time elapsed = {} s'.format(end_global - start_global))


def combine_npz():
    start = time.time()
    for dirs in os.listdir(npz_path):
        flair, t1ce, t1, t2, seg = [], [], [], [], []
        file_path = os.path.join(npz_path, dirs)

        # combining flair data from all files into one npz (in fewer parts than original if less RAM)
        print('Combining {} flair data'.format(dirs))
        for i, files in enumerate(os.listdir(file_path)):
            lis = np.load(os.path.join(file_path, files), allow_pickle=True)['arr_0']
            flair.append(lis[0])
            del lis
            if i % 20 == 0 and i != 0:
                print('Combined {} {} flair files'.format(i, dirs))
        flair_save_path = os.path.join(npz_path, dirs + '_flair_.npz')
        np.savez_compressed(flair_save_path, flair)
        print('Combined {} flair data\n'.format(dirs))
        del flair

        # combining t1ce data from all files into one npz
        print('Combining {} t1ce data'.format(dirs))
        for i, files in enumerate(os.listdir(file_path)):
            lis = np.load(os.path.join(file_path, files), allow_pickle=True)['arr_0']
            t1ce.append(lis[1])
            del lis
            if i % 20 == 0 and i != 0:
                print('Combined {} {} t1ce files'.format(i, dirs))
        t1ce_save_path = os.path.join(npz_path, dirs + '_t1ce.npz')
        np.savez_compressed(t1ce_save_path, t1ce)
        print('Combined {} t1ce data\n'.format(dirs))
        del t1ce

        # combining t1 data from all files into one npz
        print('Combining {} t1 data'.format(dirs))
        for i, files in enumerate(os.listdir(file_path)):
            lis = np.load(os.path.join(file_path, files), allow_pickle=True)['arr_0']
            t1.append(lis[2])
            del lis
            if i % 20 == 0 and i != 0:
                print('Combined {} {} t1 files'.format(i, dirs))
        t1_save_path = os.path.join(npz_path, dirs + '_t1.npz')
        np.savez_compressed(t1_save_path, t1)
        print('Combined {} t1 data\n'.format(dirs))
        del t1

        # combining t2 data from all files into one npz
        print('Combining {} t2 data'.format(dirs))
        for i, files in enumerate(os.listdir(file_path)):
            lis = np.load(os.path.join(file_path, files), allow_pickle=True)['arr_0']
            t2.append(lis[3])
            del lis
            if i % 20 == 0 and i != 0:
                print('Combined {} {} t2 files'.format(i, dirs))
        t2_save_path = os.path.join(npz_path, dirs + '_t2.npz')
        np.savez_compressed(t2_save_path, t2)
        print('Combined {} t2 data\n'.format(dirs))
        del t2

        # combining seg data from all files into one npz in case of train_data
        if dirs == 'train':
            print('Combining {} seg data'.format(dirs))
            for i, files in enumerate(os.listdir(file_path)):
                lis = np.load(os.path.join(file_path, files), allow_pickle=True)['arr_0']
                seg.append(lis[4])
                del lis
                if i % 20 == 0 and i != 0:
                    print('Combined {} {} seg files'.format(i, dirs))
            seg_save_path = os.path.join(npz_path, dirs + '_seg.npz')
            np.savez_compressed(seg_save_path, seg)
            print('Combined {} seg data\n'.format(dirs))
            del seg

    end = time.time()
    print('Combined train and validation data | Time elapsed = {} s'.format(end - start))


def predict_save():
    subject = 'BraTS19_TCIA03_216_1'
    mode = 'validation'
    subject_npz = np.load(os.path.join(npz_path, mode + '/' + subject + '.npz'), allow_pickle=True)['arr_0']

    flair = subject_npz[0]
    t1 = subject_npz[1]
    t2 = subject_npz[2]
    t1ce = subject_npz[3]

    x = np.zeros((2, 240, 240), np.float32)
    x[:1, :, :] = flair[90]
    x[1:, :, :] = t2[90]
    x = np.expand_dims(x, axis=0)

    print(x.shape)

    model = unet_9(240, 1e-4)
    model.load_weights('model/full.h5')

    pred_full = model.predict(x)

    print(pred_full.shape)

    '''
    x = np.zeros((1, 2, 240, 240), np.float32)
    x[:, :1, :, :] = flair[89:90, :, :, :]
    x[:, 1:, :, :] = t2[89:90, :, :, :]

    print(x.shape)

    model = unet_model(240, 1e-4)
    model.load_weights('model/weights-full-best.h5')

    pred_full = model.predict(x)

    print(pred_full.shape)

    imageio.imwrite('data/flair.png', flair[89:90, :, :, :][0][0])
    imageio.imwrite('data/pred_full.png', pred_full[0][0])
    '''

    '''
    imageio.imwrite('data/flair.png', flair[89:90, :, :, :][0][0])
    imageio.imwrite('data/t1.png', t1[89:90, :, :, :][0][0])
    imageio.imwrite('data/t2.png', t2[89:90, :, :, :][0][0])
    imageio.imwrite('data/t1ce.png', t1ce[89:90, :, :, :][0][0])
    '''


def debug():
    '''
    train_flair = np.load(os.path.join(npz_path, 'train_flair.npz'))['arr_0']
    train_t1ce = np.load(os.path.join(npz_path, 'train_t1ce.npz'))['arr_0']
    train_t1 = np.load(os.path.join(npz_path, 'train_t1.npz'))['arr_0']
    train_t2 = np.load(os.path.join(npz_path, 'train_t2.npz'))['arr_0']
    train_seg = np.load(os.path.join(npz_path, 'train_seg.npz'))['arr_0']
    validation_flair = np.load(os.path.join(npz_path, 'validation_flair.npz'))['arr_0']
    validation_t1ce = np.load(os.path.join(npz_path, 'validation_t1ce.npz'))['arr_0']
    validation_t1 = np.load(os.path.join(npz_path, 'validation_t1.npz'))['arr_0']
    validation_t2 = np.load(os.path.join(npz_path, 'validation_t2.npz'))['arr_0']
    print(train_flair.shape)
    print(train_t1ce.shape)
    print(train_t1.shape)
    print(train_t2.shape)
    print(train_seg.shape)
    print('=====================')
    print(validation_flair.shape)
    print(validation_t1ce.shape)
    print(validation_t1.shape)
    print(validation_t2.shape)
    '''


if __name__ == '__main__':
    # create_names_list()
    # save_all_npz()
    # combine_npz()
    # debug()
    predict_save()
    pass
