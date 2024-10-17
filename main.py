import random
from numpy.random import choice

#random.seed(1)

def main(number_of_ants, evaporation_rate):
    #Setting constants for run (Note: should be editted for different runs)
    NUMBER_OF_ITEMS = 10
    NUMBER_OF_BINS = 2
    MAX_ITERATIONS = 10000
    NUMBER_OF_ANTS = number_of_ants
    EVAPORATION_RATE = evaporation_rate
    
    #Setting lists to be used
    pheromone_matrix = [[random.random() for i in range(NUMBER_OF_BINS)] for j in range(NUMBER_OF_ITEMS)]
    BINS_MATRIX = list(range(NUMBER_OF_BINS))
    #WEIGHTS_OF_ITEMS = list(range(1, NUMBER_OF_ITEMS+1))
    WEIGHTS_OF_ITEMS = list(map(lambda i: (i**2)/2, list(range(1, NUMBER_OF_ITEMS+1))))
    print(WEIGHTS_OF_ITEMS)

    #setting_variables to be used
    number_iterations = 0

    #Running run
    while number_iterations < MAX_ITERATIONS:
        selected_paths, fitnesses = traverse_graph(NUMBER_OF_ITEMS, NUMBER_OF_BINS, NUMBER_OF_ANTS, BINS_MATRIX, WEIGHTS_OF_ITEMS, pheromone_matrix)
        add_pheromone(NUMBER_OF_ITEMS, NUMBER_OF_ANTS, pheromone_matrix, selected_paths, fitnesses)
        evaporate_pheromone(NUMBER_OF_ITEMS, NUMBER_OF_BINS, pheromone_matrix, EVAPORATION_RATE)
        #print("------------------------")
        number_iterations += 1
    print(selected_paths)
    print(fitnesses)

def traverse_graph(number_of_items, number_of_bins, number_of_ants, bins_matrix, weights_of_items, pheromone_matrix):
    #Creating selected_paths and fitnesses lists
    selected_paths = [[-1]*number_of_items for i in range(number_of_ants)]
    fitnesses = [-1]*number_of_ants

    #Generating path for each ant
    for i in range(number_of_ants):
        bin_weights = [0]*number_of_bins
        for j in range(len(pheromone_matrix)):
            #calculating probabilities from weights
            probabilities = []
            for k in range(number_of_bins):
                probabilities.append(pheromone_matrix[j][k]/sum(pheromone_matrix[j]))

            #making weighted random choice for next bin
            bin_for_item = choice(bins_matrix, 1, p=probabilities)[0]
            selected_paths[i][j] = bin_for_item
            bin_weights[bin_for_item] += weights_of_items[j]
        
        #calculating fitness for ant's path
        fitnesses[i] = max(bin_weights) - min(bin_weights)
    return selected_paths, fitnesses
        
def add_pheromone(number_of_items, number_of_ants, pheromone_matrix, selected_paths, fitnesses):
    #updating pheromone for all ants
    for i in range(number_of_ants):
        #reward function
        reward = 100/fitnesses[i]
        #adding reward to all paths taken by ant
        for j in range(number_of_items):
            pheromone_matrix[j][selected_paths[i][j]] += reward

def evaporate_pheromone(number_of_items, number_of_bins, pheromomne_matrix, evaporation_rate):
    for i in range(number_of_items):
        for j in range(number_of_bins):
            pheromomne_matrix[i][j] *= evaporation_rate

repeats = int(input("Simulation Runs: "))
number_of_ants = int(input("Number of Ants: "))
evaporation_rate = float(input("Evaporation Rate: "))
for i in range(repeats):
    main(number_of_ants, evaporation_rate)