import numpy as np
import random

def main(number_of_ants, evaporation_rate):
    best_fitness = float("inf")
    best_path = []

    #Setting constants for run (Note: should be editted for different runs)
    NUMBER_OF_ITEMS = 500
    NUMBER_OF_BINS = 10
    MAX_ITERATIONS = 10000
    
    #Setting lists to be used
    pheromone_matrix = {}
    for item in range(NUMBER_OF_ITEMS):
        for bin in range(NUMBER_OF_BINS):
            pheromone_matrix[(item, bin)] = random.uniform(0.001, 0.999)
    
    #different weights for items
    WEIGHTS_OF_ITEMS = np.arange(1, NUMBER_OF_ITEMS + 1)
    #WEIGHTS_OF_ITEMS = np.array(list(map(lambda i: (i**2)/2, range(1, NUMBER_OF_ITEMS+1))))
    
    #setting_variables to be used
    evaluations = 0

    #Running run
    while evaluations <= MAX_ITERATIONS:
        selected_paths, fitnesses = traverse_graph(NUMBER_OF_ITEMS, NUMBER_OF_BINS, WEIGHTS_OF_ITEMS, number_of_ants, pheromone_matrix)
        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_path = selected_paths[fitnesses.index(min(fitnesses))]

        if evaluations % 500 == 0:
            print("For evaluation {0} the best fitness was {1}. The average fitness was {2}".format(evaluations, min(fitnesses), sum(fitnesses)/len(fitnesses)))
        pheromone_matrix = add_pheromone(NUMBER_OF_ITEMS,NUMBER_OF_BINS, pheromone_matrix, selected_paths, fitnesses)
        pheromone_matrix = evaporate_pheromone(pheromone_matrix, evaporation_rate)
        evaluations += number_of_ants
    
    return best_fitness, best_path
    
def traverse_graph(number_of_items, number_of_bins, weights_of_items, number_of_ants, pheromone_matrix):
    #Creating selected_paths and fitnesses lists
    selected_paths = [[-1]*number_of_items for i in range(number_of_ants)]
    fitnesses = [-1]*number_of_ants

    #Calculating total pheromone probabilities
    probabilities = [[-1]*number_of_bins for i in range(number_of_items)]
    for i in range(number_of_items):
        pheromone_sum = 0
        for j in range(number_of_bins):
            pheromone_sum += pheromone_matrix[(i,j)]
        for k in range(number_of_bins):
            probabilities[i][k] = pheromone_matrix[(i, k)]/pheromone_sum

    #Generating path for each ant
    for i in range(number_of_ants):
        bin_weights = np.zeros(number_of_bins)
        for j in range(number_of_items):
            #making weighted random choice for next bin
            bin_for_item = np.random.choice(range(number_of_bins), p=probabilities[j])
            selected_paths[i][j] = bin_for_item
            bin_weights[bin_for_item] += weights_of_items[j]
        
        #calculating fitness for ant's path
        fitnesses[i] = max(bin_weights) - min(bin_weights)
    return selected_paths, fitnesses
        
def add_pheromone(number_of_items, number_of_bins, pheromone_matrix, selected_paths, fitnesses):
    for i in range(number_of_ants):
        #reward function handles solution found by setting all other paths to 0 and 0 path to 1
        if fitnesses[i] == 0:
            for j in range(number_of_items):
                for k in range(number_of_bins):
                    if k == selected_paths[i][j]:
                        pheromone_matrix[(j, k)] = 1
                    else:
                        pheromone_matrix[(j, k)] = 0
            return pheromone_matrix
        reward = 100/fitnesses[i]
        #adding reward to all paths taken by ant
        for j in range(number_of_items):
            pheromone_matrix[(j, selected_paths[i][j])] += reward
    return pheromone_matrix

def evaporate_pheromone(pheromone_matrix, evaporation_rate, min_pheromone=1e-6):
    # Evaporates the pheromone matrix
    for key in pheromone_matrix.keys():
        pheromone_matrix[key] *= evaporation_rate
    return pheromone_matrix

repeats = int(input("Simulation Runs: "))
number_of_ants = int(input("Number of Ants: "))
evaporation_rate = float(input("Evaporation Rate: "))
best_fitness = float("inf")
best_path = []

for i in range(repeats):
    #Ensures a different seed for each repeat
    print("__________________________NEW RUN_______________________________")
    np.random.seed(i)
    random.seed(i)
    run_best_fitness, run_best_path = main(number_of_ants, evaporation_rate)
    if run_best_fitness < best_fitness:
        best_fitness = run_best_fitness
        best_path = run_best_path

print("Out of all {0} runs, the lowest difference in weight between the max and min bins was {1}. The path is shown below".format(repeats, best_fitness))
print("chart showing best fitness allocation")
print(best_path)