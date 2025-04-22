import numpy as np
from random import uniform
import time
import math




def levyFlight(d):
    beta = 3 / 2
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / abs(v)** (1 / beta)
    o = step
    return o


def exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound):
    r = np.random.rand()
    # phase 1
    if  abs(F)<0.5:
        if r<p2:
            A=Best_vulture1_X-((Best_vulture1_X*current_vulture_X)/(Best_vulture1_X-current_vulture_X**2))*F
            B=Best_vulture2_X-((Best_vulture2_X*current_vulture_X)/(Best_vulture2_X-current_vulture_X**2))*F
            current_vulture_X=(A+B)/2
        else:
            current_vulture_X=random_vulture_X-abs(random_vulture_X-current_vulture_X)*F*levyFlight(variables_no)

    # phase 2
    if  abs(F)>=0.5:
        if r<p3:
            current_vulture_X=(abs((2*r)*random_vulture_X-current_vulture_X))*(F+r)-(random_vulture_X-current_vulture_X)
        else:
            s1=random_vulture_X* (np.random.rand()*current_vulture_X/(2*np.pi))*np.cos(current_vulture_X)
            s2=random_vulture_X* (np.random.rand()*current_vulture_X/(2*np.pi))*np.sin(current_vulture_X)
            current_vulture_X=random_vulture_X-(s1+s2)
    return current_vulture_X


def exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
    r1 = np.random.rand()
    if r1<p1:
        current_vulture_X=random_vulture_X-(abs((2*r1)*random_vulture_X-current_vulture_X))*F
    else:
        current_vulture_X=(random_vulture_X-(F)+np.random.rand()*((upper_bound-lower_bound)*np.random.rand()+lower_bound))
    return current_vulture_X

def RouletteWheelSelection(x):
    index = np.where(np.random.rand() <= np.cumsum(x) ,1,'first')
    return index


def random_select(Best_vulture1_X,Best_vulture2_X,alpha,betha):
    probabilities = [alpha, betha]

    if (RouletteWheelSelection(probabilities) == 1):
        random_vulture_X = Best_vulture1_X
    else:
        random_vulture_X = Best_vulture2_X
    return random_vulture_X



def AVOA(X, fobj, x_min ,x_max, max_iter):

    variables_no = X.shape[1]
    upper_bound = x_max[0, :]
    lower_bound = x_min[0, :]
    convergence_curve = np.zeros((max_iter))

    # initialize Best_vulture1, Best_vulture2
    Best_vulture1_X = np.zeros((1, variables_no))
    Best_vulture1_F = float('inf')
    Best_vulture2_X = np.zeros((1, variables_no))
    Best_vulture2_F = float('inf')

    # Controlling parameter
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    alpha = 0.8
    betha = 0.2
    gamma = 2.5

    # # Main loop
    current_iter = 0 #Loop counter
    ct = time.time()
    while current_iter < max_iter:
        for i in range(X.shape[0]):
            # Calculate  the fitness  of the population
            current_vulture_X = X[i, :]
            current_vulture_F = fobj( current_vulture_X)
            # Update the first best two vultures if needed
            if current_vulture_F < Best_vulture1_F:
                Best_vulture1_F = current_vulture_F # Update the first best bulture
                Best_vulture1_X = current_vulture_X

            if (current_vulture_F > Best_vulture1_F) and (current_vulture_F < Best_vulture2_F):
                Best_vulture2_F = current_vulture_F #  Update the second best bulture
                Best_vulture2_X = current_vulture_X
        a = uniform(-2, 2) * ((np.sin((np.pi / 2) ** (current_iter / max_iter)) ** gamma) + np.cos(
            (np.pi / 2) * (current_iter / max_iter)) - 1)
        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a

        # Update the location
        for i in range(X.shape[0]):
            current_vulture_X = X[i, :] # pick the current  vulture  back  to  the population
            F = P1 * (2 * np.random.rand() - 1)
            random_vulture_X = random_select(Best_vulture1_X, Best_vulture2_X, alpha, betha)

            if np.abs(F) >= 1: # Exploration:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)
            elif np.abs(F) < 1: # Exploitation:
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2,
                                         p3, variables_no, upper_bound, lower_bound)

            X[i, :] = current_vulture_X # place the current vulture back into  the population

        convergence_curve[current_iter] = Best_vulture1_F
        current_iter = current_iter + 1
    ct = time.time() - ct

    return Best_vulture1_F[0, 0], convergence_curve, Best_vulture1_X, ct

