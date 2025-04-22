import numpy as np
import random as rn
import time

def vectorAttack(SearchAgents_no, na):
    c = 0
    vAttack = []
    while (c <= na):
        idx = round(1 + (SearchAgents_no - 1) * np.random.rand())
        vAttack.append(idx)
        c = c + 1
    return vAttack

def survival_rate(fit, min, max):
    o = np.zeros((fit.shape[0]))
    for i in range(fit.shape[0]):
        o[i] = (max - fit[0]) / (max - min)
    return o

def getBinary():
    if np.rand() < 0.5:
        val = 0
    else:
        val = 1
    return val

def findrep(val, vector):  # return 1= repeated  0= not repeated
    band = 0
    for i in range(vector.shape[1]):
        if val == vector[i]:
            band = 1
            break
    return band

def Attack(SearchAgents_no, na, Positions, r):
    sumatory = 0
    vAttack = vectorAttack(SearchAgents_no, na)
    for j in range(len(vAttack)):
        try:
            sumatory = sumatory + Positions[vAttack[j], :] - Positions[r, :]
        except:
            pass
    sumatory = sumatory / na
    return sumatory

def DO(Positions, objfun, LB, UB, Max_iter):
    SearchAgents_no, dim = Positions.shape[0], Positions.shape[1]
    lb = LB[0, :]
    ub = UB[1, :]
    P = 0.5  # Hunting or Scavenger?  rate.Seesection3.0.4, P and Qparametersanalysis
    Q = 0.7  # Group attack or persecution?
    beta1 = -2 + 4 * np.random.rand() # -2 < beta < 2 Used in Eq. 2,
    beta2 = -1 + 2 * np.random.rand()  # -1 < beta2 < 1  Used in Eq. 2, 3, and 4
    naIni = 2  # minimum number  of dingoes  that  will  attack
    naEnd = SearchAgents_no / naIni  # maximum number of  dingoes  that  will  attack
    na = round(naIni + (
                naEnd - naIni) * np.random.rand())  # number  of dingoes that will attack, used in Attack.m Section  2.2 .1: Group attack
    # Positions = initialization(SearchAgents_no, dim, ub, lb)
    Fitness = np.zeros((SearchAgents_no))
    for i in range(SearchAgents_no):
        Fitness[i] = objfun(Positions[i, :])  # get fitness
    vMin, minIdx = min(Fitness),np.argmin(Fitness)  # the  min fitness value vMin and the  position minIdx
    theBestVct = Positions[minIdx, :]  # the best vector
    vMax, maxIdx = max(Fitness),np.argmax(Fitness)  # the max fitness value vMax and the position maxIdx
    Convergence_curve = np.zeros((Max_iter))
    Convergence_curve[0] = vMin
    survival = survival_rate(Fitness, vMin, vMax)  # Section  2.2 .4 Dingoes'survival rates
    v = - np.zeros((SearchAgents_no))
    ct = time.time()
    # Main loop
    for t in range(Max_iter):
        for r in range(SearchAgents_no):
            if np.random.rand() < P:  # If Hunting?
                sumatory = Attack(SearchAgents_no, na, Positions, r)  # Section 2.2.1, Strategy1: Part of  Eq .2
                if np.random.rand() < Q:  # If group attack?
                    v[r] =( beta1 * sumatory - theBestVct)[0]  # Strategy 1: Eq .2
                else:  # Persecution
                    r1 = round(1 + (SearchAgents_no - 1) * np.random.rand())  #
                    try:
                        v[r] = (theBestVct + beta1 * (np.exp(beta2)) * ((Positions[r1, :] - Positions[r, :])))[0]  # Section  2.2 .2, Strategy 2: Eq .3
                    except:
                        pass
            else:  # Scavenger
                r1 = np.round(1 + (SearchAgents_no - 1) * np.random.rand())
                r1 = int(r1)
                try:
                    v[r] =((np.exp(beta2) * Positions[r1, :] - ((-1) ^ 1) * Positions[r,:]) / 2)[0] # Section2.2.3, Strategy3: Eq.4
                except:
                    pass
            # Return  back  the search  agents  that    go  beyond   the  boundaries  of the  search  space.
            Flag4ub = v[r] > ub[0]
            Flag4lb = v[r] < lb[0]
            v[r] = (v[r] * (~(Flag4ub + Flag4lb))) + ub[0] * Flag4ub + lb[0] * Flag4lb  # Evaluate new solutions
            Fnew = objfun(v[r])  # get fitness
            # Update if the  solution improves
            if Fnew[0][0] <= Fitness[r]:
                Positions[r, :] = v[r]
                Fitness[r] = Fnew[0][0]
            if Fnew[0][0] <= vMin:
                theBestVct = v[r]
                vMin = Fnew
        Convergence_curve[t] = vMin
        ct = time.time() - ct
    return vMin,Convergence_curve, theBestVct,  ct

