import os
import random as rn
import numpy as np
import cv2 as cv
from numpy import matlib
from AVOA import AVOA
from AZOA import AZOA
from Batch_Split import Batch_Split
from DO import DO
from EOO import EOO
from Glob_Vars import Glob_Vars
from Image_Reults import Image_Results1
from Model_AM_EFFIECIENTNET import Model_AM_EFFIECIENTNET
from Model_CNN2 import Model_CNN
from Model_EFFICIENTNET import Model_EFFICIENTNET
from Model_FCM import Model_FCM
from Model_GRU import Model_GRU
from Model_Trans_Resunetpp import Train_3d_trans_resunetp
from Model_Trans_Resunetpp import Train_3d_trans_resunetpp
from PROPOSED import PROPOSED
from Plot_Results import Plot_Results, plot_Method, Confusion_matrix, Plot_ROC,  Plot_Fitness
from TransResUnetplusplus import TransResUnetplusplus
from objective_function import Objfun

# Read Dataset
an =0
if an == 1:
    images = []
    dir = './Dataset/'
    dir1 = os.listdir(dir)
    for i in range(len(dir1)):
        file = dir + dir1[i]
        read = cv.imread(file)
        read = cv.resize(read, [700, 700])
        images.append(read)
    np.save('Original.npy',images)

# Read Ground_Truth
an = 0
if an == 1:
    dir = './Dataset/'
    dir1 = os.listdir(dir)
    imgs=[]
    imgs1 = []
    for i in range(len(dir1)):
        file = dir+dir1[i]
        read = cv.imread(file)
        read = cv.cvtColor(read,cv.COLOR_RGB2GRAY)
        read = cv.resize(read,[700,700])
        imgs.append(read)
    imgs = np.asarray(imgs)
    feat = Model_FCM(imgs)
    np.save('Ground_Truth.npy',feat)

# Read Target
an = 0
if an == 1:
    targets =[]
    for i in range(4):
        Images = np.load('Ground_Truth.npy', allow_pickle=True)[i]
        patche = Batch_Split(Images)
        Target = []
        for j in range(len(patche)):
            gr_tru = patche[j]
            uniq = np.unique(gr_tru)
            lenUniq = [len(np.where(gr_tru == uniq[k])[0]) for k in range(len(uniq))]
            maxIndex = np.where(lenUniq == np.max(lenUniq))[0][0]
            target = uniq[maxIndex]
            Target.append(target)
        Targ = np.asarray(Target)
        uni = np.unique(Targ)
        tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
        for a in range(len(uni)):
            ind = np.where((Targ == uni[a]))
            tar[ind[0], i] = a
        targets.append(tar)
    np.save('Targets.npy', targets)

# Segmentation
an = 0
if an == 1:
    Images = np.load('Original.npy',allow_pickle=True)
    Gt = np.load('Ground_Truth.npy',allow_pickle=True)
    Seg = Train_3d_trans_resunetp(Images,Gt,Images)
    np.save('Segmented.npy',Seg)

# Optimization for Prediction
an = 0
if an == 1:
    seg = np.load('Segmented.npy', allow_pickle=True)
    Targets = np.load('Target.npy', allow_pickle=True)
    Glob_Vars.seg = seg
    Glob_Vars.Tar = Targets
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat([5, 5,50], Npop, 1)
    xmax = matlib.repmat([255, 50,250], Npop, 1)
    fname = Objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 25
    print("DHOA...")
    [bestfit1, fitness1, bestsol1, time1] = DO(initsol, fname, xmin, xmax, Max_iter)

    print("HHO...")
    [bestfit2, fitness2, bestsol2, time2] = EOO(initsol, fname, xmin, xmax, Max_iter)

    print("BOA...")
    [bestfit4, fitness4, bestsol4, time3] = AVOA(initsol, fname, xmin, xmax, Max_iter)

    print("RSO...")
    [bestfit3, fitness3, bestsol3, time4] = AZOA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    Bestsol_Feat = ([bestsol1,bestsol2,bestsol3,bestsol4,bestsol5])
    Fitness = ([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('BestSol.npy', Bestsol_Feat)
    np.save('Fitness.npy', Fitness)

## Segmentation
an = 0
if an == 1:
    Epochs = [100, 200, 300, 400,500]
    Eval_all = []
    Feat = np.load('Segmented.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy', allow_pickle=True)
    Targets =np.load('Target.npy',allow_pickle = True)
    for i in range(len(Epochs)):
        Eval = np.zeros((10, 600))
        learnper = round(Targets.shape[0] * Epochs[i])
        for j in range(sol.shape[0]):
            learnper = round(Feat.shape[0] * 0.75)
            train_data = Feat[learnper:, :]
            train_target = Targets[learnper:, :]
            test_data = Feat[:learnper, :]
            test_target = Targets[:learnper, :]
            Eval = Model_AM_EFFIECIENTNET(train_data, train_target, test_data, test_target, sol[j].astype('int'))
        Train_Data1 = Feat[learnper:, :]
        Test_Data1 = Feat[:learnper, :]
        Train_Target = Targets[learnper:, :]
        Test_Target = Targets[:learnper, :]
        Eval[5, :] = Model_CNN(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[6, :] = Model_GRU(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[7, :] = Model_EFFICIENTNET(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[8, :] = Model_AM_EFFIECIENTNET(Train_Data1, Train_Target, Test_Data1, Test_Target)
        Eval[9, :] = Eval[4, :]
        Eval_all.append(Eval)
    np.save('Eval_all.npy', np.asarray(Eval_all))




Plot_Results()
plot_Method()
Confusion_matrix()
Plot_ROC()
Plot_Fitness()
Image_Result()
Image_Results1()