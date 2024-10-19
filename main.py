import numpy as np

def main(number_of_ants, evaporation_rate):
    #Setting constants for run (Note: should be editted for different runs)
    NUMBER_OF_ITEMS = 500
    NUMBER_OF_BINS = 10
    MAX_ITERATIONS = 10000
    NUMBER_OF_ANTS = number_of_ants
    EVAPORATION_RATE = evaporation_rate
    
    #Setting lists to be used
    pheromone_matrix = np.random.rand(NUMBER_OF_ITEMS, NUMBER_OF_BINS)
    BINS_MATRIX = np.arange(NUMBER_OF_BINS)
    
    #different weights for items
    #WEIGHTS_OF_ITEMS = np.arange(1, NUMBER_OF_ITEMS + 1)
    WEIGHTS_OF_ITEMS = np.array(list(map(lambda i: (i**2)/2, range(1, NUMBER_OF_ITEMS+1))))
    
    #setting_variables to be used
    evaluations = 0

    #Running run
    while evaluations < MAX_ITERATIONS:
        #print(evaluations)
        selected_paths, fitnesses = traverse_graph(NUMBER_OF_ITEMS, NUMBER_OF_BINS, NUMBER_OF_ANTS, BINS_MATRIX, WEIGHTS_OF_ITEMS, pheromone_matrix)
        pheromone_matrix = add_pheromone(NUMBER_OF_ITEMS, NUMBER_OF_ANTS, pheromone_matrix, selected_paths, fitnesses)
        pheromone_matrix = evaporate_pheromone(pheromone_matrix, EVAPORATION_RATE)
        #print("------------------------")
        evaluations += number_of_ants
    
    
def traverse_graph(number_of_items, number_of_bins, number_of_ants, bins_matrix, weights_of_items, pheromone_matrix):
    #Creating selected_paths and fitnesses lists
    selected_paths = [[-1]*number_of_items for i in range(number_of_ants)]
    fitnesses = [-1]*number_of_ants

    #Generating path for each ant
    for i in range(number_of_ants):
        bin_weights = np.zeros(number_of_bins)
        for j in range(number_of_items):
            pheromone_sum = np.sum(pheromone_matrix[j])
            probabilities = pheromone_matrix[j]/pheromone_sum

            #making weighted random choice for next bin
            bin_for_item = np.random.choice(bins_matrix, p=probabilities)
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
    return pheromone_matrix

def evaporate_pheromone(pheromone_matrix, evaporation_rate):
    pheromone_matrix *= evaporation_rate
    return pheromone_matrix

def evaluate_solution(pheromone_matrix):
    print("evalutating path")

repeats = int(input("Simulation Runs: "))
number_of_ants = int(input("Number of Ants: "))
evaporation_rate = float(input("Evaporation Rate: "))

for i in range(repeats):
    #Ensures a different seed for each repeat
    np.random.seed(i)  
    main(number_of_ants, evaporation_rate)