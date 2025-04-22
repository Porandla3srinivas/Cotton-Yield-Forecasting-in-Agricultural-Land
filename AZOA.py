import random
import time

import numpy as np

def AZOA(position,func, lower, upper, max_iter):
    pop_size = position.shape[0]
    population = []
    for a in range(pop_size):
        population.append((position, func(position)))
    vMin, minIdx = min(population), np.argmin(population)  # the  min fitness value vMin and the  position minIdx
    pop_fit = population[minIdx, :]  # the best vector
    ct = time.time()
    convergence_curve = np.zeros((max_iter))
    for current_iter in range(max_iter):
        # Generate new solutions.
        new_population = []
        for i in range(pop_size):
          # Generate a new solution by mutation.
          mutation = [random.uniform(-1, 1) for _ in range(len(lower))]
          new_position = [position + mutation for position in population[i][0]]
          new_position = [min(upper, max(lower, position)) for position in new_position]
          # Generate a new solution by crossover.
          crossover = random.randint(0, pop_size - 1)
          new_position = [position + population[crossover][0][j] * 0.5 for j in range(len(lower))]
          new_position = [min(upper, max(lower, position)) for position in new_position]
          # Evaluate the new solutions.
          new_value = func(new_position)
          # Select the best solution.
          if new_value < population[i][1]:
            new_population.append((new_position, new_value))
          else:
            new_population.append(population[i])
        population = new_population
        convergence_curve[current_iter] = population
    best_solution = min(population, key=lambda x: x[1])

    ct = time.time()-ct
    return best_solution, convergence_curve, pop_fit , ct