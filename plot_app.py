import os
import matplotlib.pyplot as plt


def save_images_app(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, name):
    plt.figure()
    plt.axis('off')
    plt.imshow(flair[0], cmap='gray')
    plt.savefig('static/data/result/flair.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('result/{}/flair.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(t1ce[0], cmap='gray')
    plt.savefig('static/data/result/t1ce.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('result/{}/t1ce.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(t1[0], cmap='gray')
    plt.savefig('static/data/result/t1.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('result/{}/t1.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(t2[0], cmap='gray')
    plt.savefig('static/data/result/t2.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('result/{}/t2.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    if resume:
        plt.figure()
        plt.axis('off')
        plt.imshow(pred_full[0][0])
        plt.savefig('static/data/result/overall.png', bbox_inches='tight', pad_inches=0)
        plt.savefig('result/{}/overall.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure()
        plt.axis('off')
        plt.imshow(core[0])
        plt.savefig('static/data/result/core.png', bbox_inches='tight', pad_inches=0)
        plt.savefig('result/{}/core.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure()
        plt.axis('off')
        plt.imshow(et[0])
        plt.savefig('static/data/result/et.png', bbox_inches='tight', pad_inches=0)
        plt.savefig('result/{}/et.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure()
        plt.axis('off')
        plt.imshow(tmp[0])
        plt.savefig('static/data/result/full.png', bbox_inches='tight', pad_inches=0)
        plt.savefig('result/{}/full.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()


# plot flair, t1ce, t1 and t2 data
def plot_app_original_data(t1, t2, flair, t1ce, resume, name):
    # if resume:
    #     print('Plotting Original Data')
    plt.figure(figsize=(15, 10))

    plt.subplot(241)
    plt.title('Flair')
    plt.axis('off')
    plt.imshow(flair[0], cmap='gray')

    plt.subplot(242)
    plt.title('T1c')
    plt.axis('off')
    plt.imshow(t1ce[0], cmap='gray')

    plt.subplot(243)
    plt.title('T1')
    plt.axis('off')
    plt.imshow(t1[0], cmap='gray')

    plt.subplot(244)
    plt.title('T2')
    plt.axis('off')
    plt.imshow(t2[0], cmap='gray')

    plt.savefig('static/data/result/1_original_data.png')
    plt.savefig('result/{}/1_original_data.png'.format(name))


# plot initial prediction showing overall tumor
def plot_app_overall(t2, flair, pred_full, name):
    # print('Plotting Full Prediction')
    plt.figure(figsize=(15, 10))

    plt.subplot(341)
    plt.title('Flair')
    plt.axis('off')
    plt.imshow(flair[0], cmap='gray')

    plt.subplot(342)
    plt.title('T2')
    plt.axis('off')
    plt.imshow(t2[0], cmap='gray')

    plt.subplot(343)
    plt.title('prediction(full)')
    plt.axis('off')
    plt.imshow(pred_full[0][0])

    plt.savefig('static/data/result/2_overall_prediction.png')
    plt.savefig('result/{}/2_overall_prediction.png'.format(name))


# plot final prediction showing core, ET and full tumor
def plot_app_final(t1, t2, flair, t1ce, pred_full, core, et, tmp, name):
    # print('Plotting Core, ET and Full prediction\n')
    plt.figure(figsize=(15, 10))

    plt.subplot(341)
    plt.title('Flair')
    plt.axis('off')
    plt.imshow(flair[0], cmap='gray')

    plt.subplot(342)
    plt.title('T1c')
    plt.axis('off')
    plt.imshow(t1ce[0], cmap='gray')

    plt.subplot(343)
    plt.title('T1')
    plt.axis('off')
    plt.imshow(t1[0], cmap='gray')

    plt.subplot(344)
    plt.title('T2')
    plt.axis('off')
    plt.imshow(t2[0], cmap='gray')

    plt.subplot(345)
    plt.title('Prediction (Full)')
    plt.axis('off')
    plt.imshow(pred_full[0][0])

    plt.subplot(346)
    plt.title('Prediction (Core)')
    plt.axis('off')
    plt.imshow(core[0])

    plt.subplot(347)
    plt.title('Prediction (ET)')
    plt.axis('off')
    plt.imshow(et[0])

    plt.subplot(348)
    plt.title('Prediction (All)')
    plt.axis('off')
    plt.imshow(tmp[0])

    plt.savefig('static/data/result/3_final_prediction.png')
    plt.savefig('result/{}/3_final_prediction.png'.format(name))
