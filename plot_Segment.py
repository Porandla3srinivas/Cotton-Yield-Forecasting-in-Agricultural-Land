import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import cv2 as cv
from Segmentation_Evaluation import Segmentation_Evaluation

def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out

no_of_dataset=1

def Plot_Image_Results():
    matplotlib.use('TkAgg')
    eval = np.load('Eval_seg.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Dice', 'Jaccard']
    value = eval[ :, :, :]
    stat = np.zeros((value.shape[1], value.shape[2], 5))
    for j in range(value.shape[1]): # For all algms and Mtds
        for k in range(value.shape[2]): # For all terms
            stat[j, k, :] = Statistical(value[:, j, k])
    stat = stat
    for k in range(len(Terms)):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, stat[0, k, :], color='#f97306', width=0.10, label="UNet")
        ax.bar(X + 0.10, stat[1, k, :], color='#cc3f81', width=0.10, label="ResUNet")
        ax.bar(X + 0.20, stat[2, k, :], color='#ccbc3f', width=0.10, label="Trans-UNet")
        ax.bar(X + 0.30, stat[3, k, :], color='c', width=0.10, label="Trans-ResUNet")
        ax.bar(X + 0.40, stat[4, k, :], color='k', width=0.10, label="3D-Trans-ResUNet+")
        plt.xticks(X + 0.10, ('Best', 'Worst', 'Mean', 'Median', 'Std'))
        plt.ylabel(Terms[k])
        plt.xlabel('Statistical Analysis')
        plt.legend(loc=1)
        path1 = "./Results/Segmented_Image_bar_%s.png" % (Terms[k])
        plt.savefig(path1)
        plt.show()

Plot_Image_Results()