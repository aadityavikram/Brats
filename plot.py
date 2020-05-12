import os
import matplotlib.pyplot as plt


def save_images(flair, t1ce, t1, t2, pred_full, core, et, tmp, resume, name):
    plt.figure()
    plt.axis('off')
    plt.imshow(flair[90][0], cmap='gray')
    plt.savefig('result/{}/flair.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(t1ce[90][0], cmap='gray')
    plt.savefig('result/{}/t1ce.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(t1[90][0], cmap='gray')
    plt.savefig('result/{}/t1.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(t2[90][0], cmap='gray')
    plt.savefig('result/{}/t2.png'.format(name), bbox_inches='tight', pad_inches=0)
    plt.close()

    if resume:
        plt.figure()
        plt.axis('off')
        plt.imshow(pred_full[0][0])
        plt.savefig('result/{}/overall.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure()
        plt.axis('off')
        plt.imshow(core[0])
        plt.savefig('result/{}/core.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure()
        plt.axis('off')
        plt.imshow(et[0])
        plt.savefig('result/{}/et.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure()
        plt.axis('off')
        plt.imshow(tmp[0])
        plt.savefig('result/{}/full.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()


def save_slice_images(pred_full, core, et, tmp, subject, slices):
    if not os.path.exists('result/' + subject + '_images/overall'):
        os.makedirs('result/' + subject + '_images/overall')
    if not os.path.exists('result/' + subject + '_images/core'):
        os.makedirs('result/' + subject + '_images/core')
    if not os.path.exists('result/' + subject + '_images/et'):
        os.makedirs('result/' + subject + '_images/et')
    if not os.path.exists('result/' + subject + '_images/full'):
        os.makedirs('result/' + subject + '_images/full')

    plt.figure()
    plt.axis('off')
    plt.imshow(pred_full[0][0])
    plt.savefig('result/' + subject + '_images/overall/overall_{}.png'.format(slices + 1), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(core[0])
    plt.savefig('result/' + subject + '_images/core/core_{}.png'.format(slices + 1), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(et[0])
    plt.savefig('result/' + subject + '_images/et/et_{}.png'.format(slices + 1), bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.figure()
    plt.axis('off')
    plt.imshow(tmp[0])
    plt.savefig('result/' + subject + '_images/full/full_{}.png'.format(slices + 1), bbox_inches='tight', pad_inches=0)
    plt.close()


# plot flair, t1ce, t1 and t2 data
def plot_original_data(t1, t2, flair, t1ce, resume, src):
    if resume:
        print('Plotting Original Data')
    plt.figure(figsize=(15, 10))

    plt.subplot(241)
    plt.title('Flair')
    plt.axis('off')
    plt.imshow(flair[90][0], cmap='gray')

    plt.subplot(242)
    plt.title('T1c')
    plt.axis('off')
    plt.imshow(t1ce[90][0], cmap='gray')

    plt.subplot(243)
    plt.title('T1')
    plt.axis('off')
    plt.imshow(t1[90][0], cmap='gray')

    plt.subplot(244)
    plt.title('T2')
    plt.axis('off')
    plt.imshow(t2[90][0], cmap='gray')

    plt.savefig(src + '1_original_data.png')
    if resume:
        plt.show()


# plot initial prediction showing overall tumor
def plot_overall(t2, flair, pred_full, src):
    print('Plotting Full Prediction')
    plt.figure(figsize=(15, 10))

    plt.subplot(341)
    plt.title('Flair')
    plt.axis('off')
    plt.imshow(flair[90][0], cmap='gray')

    plt.subplot(342)
    plt.title('T2')
    plt.axis('off')
    plt.imshow(t2[90][0], cmap='gray')

    plt.subplot(343)
    plt.title('prediction(full)')
    plt.axis('off')
    plt.imshow(pred_full[0][0])

    plt.savefig(src + '2_overall_prediction.png')
    plt.show()


# plot final prediction showing core, ET and full tumor
def plot_final(t1, t2, flair, t1ce, pred_full, core, et, tmp, src):
    print('Plotting Core, ET and Full prediction\n')
    plt.figure(figsize=(15, 10))

    plt.subplot(341)
    plt.title('Flair')
    plt.axis('off')
    plt.imshow(flair[90][0], cmap='gray')

    plt.subplot(342)
    plt.title('T1c')
    plt.axis('off')
    plt.imshow(t1ce[90][0], cmap='gray')

    plt.subplot(343)
    plt.title('T1')
    plt.axis('off')
    plt.imshow(t1[90][0], cmap='gray')

    plt.subplot(344)
    plt.title('T2')
    plt.axis('off')
    plt.imshow(t2[90][0], cmap='gray')

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

    plt.savefig(src + '3_final_prediction.png')
    plt.show()
