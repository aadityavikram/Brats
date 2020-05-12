import numpy as np


def crop_img(t1ce, pred, size):
    crop_x = []
    list_xy = []
    p_tmp = pred[0, :, :]
    p_tmp[p_tmp > 0.2] = 1  # threshold
    p_tmp[p_tmp != 1] = 0
    # get middle point from prediction of full tumor
    index_xy = np.where(p_tmp == 1)  # get all the axial of pixel which value is 1

    if index_xy[0].shape[0] == 0:  # skip when no tumor
        return [], []

    center_x = (max(index_xy[0]) + min(index_xy[0])) / 2
    center_y = (max(index_xy[1]) + min(index_xy[1])) / 2

    if center_x >= 176:
        center_x = center_x - 8

    length = max(index_xy[0]) - min(index_xy[0])
    width = max(index_xy[1]) - min(index_xy[1])

    if length <= 64:
        if width <= 64:  # 64x64
            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x - size / 2): int(center_x + size / 2), int(center_y - size / 2): int(center_y + size / 2)]
            crop_x.append(img_x)
            list_xy.append((int(center_x - size / 2), int(center_y - size / 2)))

        else:  # 64x128
            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x - size / 2): int(center_x + size / 2), int(center_y - size): int(center_y)]
            crop_x.append(img_x)
            list_xy.append((int(center_x - size / 2), int(center_y - size)))

            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x - size / 2): int(center_x + size / 2), int(center_y + 1): int(center_y + size + 1)]
            crop_x.append(img_x)
            list_xy.append((int(center_x - size / 2), int(center_y)))

    else:
        if width <= 64:  # 128x64
            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x - size): int(center_x), int(center_y - size / 2): int(center_y + size / 2)]
            crop_x.append(img_x)
            list_xy.append((int(center_x - size), int(center_y - size / 2)))

            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x + 1): int(center_x + size + 1), int(center_y - size / 2): int(center_y + size / 2)]
            crop_x.append(img_x)
            list_xy.append((int(center_x), int(center_y - size / 2)))

        else:  # 128x128
            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x - size): int(center_x), int(center_y - size): int(center_y)]
            crop_x.append(img_x)
            list_xy.append((int(center_x - size), int(center_y - size)))

            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x + 1): int(center_x + size + 1), int(center_y - size): int(center_y)]
            crop_x.append(img_x)
            list_xy.append((int(center_x), int(center_y - size)))

            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x - size): int(center_x), int(center_y + 1): int(center_y + size + 1)]
            crop_x.append(img_x)
            list_xy.append((int(center_x - size), int(center_y)))

            img_x = np.zeros((1, size, size), np.float32)
            img_x[:, :, :] = t1ce[:, int(center_x + 1): int(center_x + size + 1), int(center_y + 1): int(center_y + size + 1)]
            crop_x.append(img_x)
            list_xy.append((int(center_x), int(center_y)))

    return np.array(crop_x), list_xy  # (y,x)


def color(pred_full, pred_core, pred_et, li):  # input image is [n,1, y, x]
    # first put the pred_full on T1c
    pred_full[pred_full > 0.2] = 2  # 240x240
    pred_full[pred_full != 2] = 0
    pred_core[pred_core > 0.2] = 1  # 64x64
    pred_core[pred_core != 1] = 0
    pred_et[pred_et > 0.2] = 4  # 64x64
    pred_et[pred_et != 4] = 0

    total = np.zeros((1, 240, 240), np.float32)
    total[:, :, :] = pred_full[:, :, :]
    for i in range(pred_core.shape[0]):
        for j in range(64):
            for k in range(64):
                if pred_core[i, 0, j, k] != 0 and pred_full[0, li[i][0] + j, li[i][1] + k] != 0:
                    total[0, li[i][0] + j, li[i][1] + k] = pred_core[i, 0, j, k]
                if pred_et[i, 0, j, k] != 0 and pred_full[0, li[i][0] + j, li[i][1] + k] != 0:
                    total[0, li[i][0] + j, li[i][1] + k] = pred_et[i, 0, j, k]

    return total
