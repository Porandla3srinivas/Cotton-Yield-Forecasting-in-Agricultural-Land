import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def Image_Results1():
    Orig = np.load('Original.npy',allow_pickle=True)
    # Images = np.load('Pre-Processed_Image.npy', allow_pickle=True)
    # segment = np.load('Segmented.npy', allow_pickle=True)
    grnd = np.load('Ground_Truth.npy', allow_pickle=True)
    for j in range(4):
        original = Orig[j]
        gt = grnd[j]
        # ori = cv.cvtColor(images[j], cv.COLOR_RGB2GRAY)
        orig = cv.resize(original, [512, 512])
        seg = cv.resize(gt, [512, 512])
        # lb = cv.resize(lbp[j], [256, 256])
        # lv = cv.resize(lvp[j], [256, 256])
        # extr = cv.resize(ext[j], [256, 256])
        # ft = cv.add(original, gt)
        # hist = cv.equalizeHist(ft)
        im = np.concatenate((orig,seg), axis=1)
        cv.imshow('                                                                     Original                                                                                                                                                  Segmented', im)
        cv.waitKey(0)
        # path1 = "./Results/image/image_%s.png" % (j + 1)
        # plt.savefig(path1)
        # original = Orig[j]

        # seg = segment[j]

        # # cv.imshow('im', Output)
        # # cv.waitKey(0)
        # gt = grnd[j]
        # fig, ax = plt.subplots(1, 3)
        # plt.suptitle("Image %d" % (j + 1), fontsize=20)
        # plt.subplot(1, 2, 1)
        # plt.title('Orig')
        # plt.imshow(original)
        # plt.subplot(1, 2, 2)
        # plt.title('Segmented')
        # plt.imshow(gt)
        # # plt.subplot(1, 3, 3)
        # # plt.title('Segmented')
        # # plt.imshow(seg)
        # path1 = "./Results/image/Dataset_%simage.png" % (j + 1)
        # plt.savefig(path1)
        # plt.show()
        cv.imwrite('./Results/image/orig-' + str(j + 1) + '.png', orig)
        cv.imwrite('./Results/image/seg-' + str(j + 1) + '.png', seg)
        # # cv.imwrite('./Results/segment-' + str(j + 1) + '.png', seg)
        # # cv.imwrite('./Results/ground-' + str(j + 1) + '.png', gt)

# Image_Results1()