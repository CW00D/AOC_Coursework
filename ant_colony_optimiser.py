import numpy as np
import random
import os
import csv

class Node:
    def __init__(self, number_of_bins: int, item_number: int, bin_number: int):
        #Node attributes
        self.linked_nodes = []
        self.linked_node_probabilities = [random.uniform(0.001, 0.999)]*number_of_bins
        self.bin_number = bin_number
        self.item_number = item_number

    def set_linked_nodes(self, nodes):
        self.linked_nodes = nodes

    def add_pheromone(self, index_to_add_pheromone, pheromone_to_add):
        self.linked_node_probabilities[index_to_add_pheromone] += pheromone_to_add

    def evaporate_pheromone(self, evaporation_rate: float):
        self.linked_node_probabilities = [probability*evaporation_rate for probability in self.linked_node_probabilities]

def setup_pheromone_graph(number_of_items, number_of_bins):
    #creating all nodes
    nodes_matrix = []
    nodes_list = []
    for i in range(number_of_items):
        temp_array = []
        for j in range(number_of_bins):
            new_node = Node(number_of_bins, i, j)
            temp_array.append(new_node)
            nodes_list.append(new_node)
        nodes_matrix.append(temp_array)
    
    start_node = Node(number_of_bins, -1, -1)
    start_node.set_linked_nodes(nodes_matrix[0])
    nodes_list.insert(0, start_node)

    #adding linked nodes to nodes
    for i in range(number_of_items-1):
        for node in nodes_matrix[i]:
            node.set_linked_nodes(nodes_matrix[i+1])    

    return nodes_list
    
def weighted_choice(probabilities):
    total = sum(probabilities)
    normalized_probabilities = [p / total for p in probabilities]
    return np.random.choice(len(probabilities), p=normalized_probabilities)

def run_simulation(experiment_number, experiment_set, run_number, number_of_ants, evaporation_rate):
    #Setting constants for run (Note: should be editted for different runs)
    NUMBER_OF_ITEMS = 500
    MAX_ITERATIONS = 10000
    
    #Different experiment setups
    if experiment_number == 1:
        WEIGHTS_OF_ITEMS = np.arange(1, NUMBER_OF_ITEMS + 1)
        NUMBER_OF_BINS = 10
        log_file = "BPP1_Experiment_Results.csv"
    elif experiment_number == 2:
        WEIGHTS_OF_ITEMS = np.array(list(map(lambda i: (i**2)/2, range(1, NUMBER_OF_ITEMS+1))))
        NUMBER_OF_BINS = 50
        log_file = "BPP2_Experiment_Results.csv"

    #setting up the log file for experiment
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Experiment_Set", "Run_Number", "Iteration", "p", "e", "Best_Fitness", "Average_Fitness", "Standard_Deviation"])

    nodes_list = setup_pheromone_graph(NUMBER_OF_ITEMS, NUMBER_OF_BINS)

    #Setting_variables to be used
    evaluations = 0

    #Running run
    while evaluations <= MAX_ITERATIONS:
        selected_paths, fitnesses = traverse_graph(NUMBER_OF_ITEMS, NUMBER_OF_BINS, number_of_ants, nodes_list, WEIGHTS_OF_ITEMS)
        add_pheromone(nodes_list, nodes_list[0], selected_paths, fitnesses)
        evaporate_pheromone(nodes_list, evaporation_rate)

        #Logging result to experiment results
        if evaluations % 500 == 0:
            fitness_standard_deviation = np.std(fitnesses)
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([experiment_set, run_number, evaluations, number_of_ants, evaporation_rate, min(fitnesses), sum(fitnesses)/len(fitnesses), fitness_standard_deviation])
        
        evaluations += number_of_ants
    
def traverse_graph(number_of_items, number_of_bins, number_of_ants, nodes_list, item_weights):
    #Creating selected_paths and fitnesses lists
    selected_paths = [[]*number_of_items for i in range(number_of_ants)]
    fitnesses = [-1]*number_of_ants

    #Generating path for each ant
    for i in range(number_of_ants):
        bin_weights = [0]*number_of_bins
        current_node = nodes_list[0]
        while current_node.linked_nodes != []:
            #Take path decision
            probabilities = current_node.linked_node_probabilities
            next_node_index = weighted_choice(probabilities)
            current_node = current_node.linked_nodes[next_node_index]

            #Add node to selected path for ant and add items weight to bin
            selected_paths[i].append(current_node)
            bin_weights[current_node.bin_number] += item_weights[current_node.item_number]

        #Calculate fitness for selected path
        fitnesses[i] = max(bin_weights) - min(bin_weights)
    
    return selected_paths, fitnesses

def add_pheromone(nodes_list, start_node, selected_paths, fitnesses):
    for i in range(len(selected_paths)):
        pheromone_to_add = 100/fitnesses[i]
        previous_node = start_node
        for node in selected_paths[i]:
            index_to_add_pheromone_to = previous_node.linked_nodes.index(node)
            previous_node.add_pheromone(index_to_add_pheromone_to, pheromone_to_add)
            previous_node = node
            
def evaporate_pheromone(nodes_list, evaporation_rate):
    for node in nodes_list:
        node.evaporate_pheromone(evaporation_rate)

def run_experiment(experiment_number, experiment_set, repeats, number_of_ants, evaporation_rate):
    for run_number in range(repeats):
        np.random.seed(run_number)
        random.seed(run_number)
        run_simulation(experiment_number, experiment_set, run_number, number_of_ants, evaporation_rate)
    
if __name__ == "__main__":
    #Experiment 1
    run_experiment(1, 1, 5, 100, 0.9)
    run_experiment(1, 2, 5, 100, 0.6)
    run_experiment(1, 3, 5, 10, 0.9)
    run_experiment(1, 4, 5, 10, 0.6)

    #Experiment 2
    run_experiment(2, 1, 5, 100, 0.9)
    run_experiment(2, 2, 5, 100, 0.6)
    run_experiment(2, 3, 5, 10, 0.9)
    run_experiment(2, 4, 5, 10, 0.6)