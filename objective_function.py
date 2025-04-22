import numpy as np
from Glob_Vars import Glob_Vars
from Model_AM_EFFIECIENTNET import Model_AM_EFFICIENTNET


def Objfun(Soln):

    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)
        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        seg = Glob_Vars.seg
        Tar = Glob_Vars.Tar
        learnper = round(seg.shape[0] * 0.75)
        train_data = seg[learnper:, :]
        train_target = Tar[learnper:, :]
        test_data = seg[:learnper, :]
        test_target = Tar[:learnper, :]
        Eval = Model_AM_EFFICIENTNET(train_data, train_target, test_data, test_target, sol.astype('int'))
        Fitn[i] = 1 / (Eval[4])

    return Fitn

