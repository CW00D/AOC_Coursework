import numpy as np
import random
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Node Class
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

#Helper Methods
def weighted_choice(probabilities):
    total = np.sum(probabilities)
    normalized_probabilities = probabilities / total
    random_value = np.random.random()
    cumulative_sum = 0.0
    for i in range(len(normalized_probabilities)):
        cumulative_sum += normalized_probabilities[i]
        if random_value < cumulative_sum:
            return i
    return len(normalized_probabilities) - 1

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
    
#Experiment Methods
def run_experiment(experiment_number, experiment_set, repeats, number_of_ants, evaporation_rate):
    for run_number in range(repeats):
        np.random.seed(run_number)
        random.seed(run_number)
        run_simulation(experiment_number, experiment_set, run_number, number_of_ants, evaporation_rate)

def run_simulation(experiment_number, experiment_set, run_number, number_of_ants, evaporation_rate):
    #Setting constants for run (Note: should be editted for different runs)
    NUMBER_OF_ITEMS = 500
    MAX_ITERATIONS = 10000
    
    #Different experiment setups
    if experiment_number == 1:
        WEIGHTS_OF_ITEMS = np.arange(1, NUMBER_OF_ITEMS + 1)
        NUMBER_OF_BINS = 10
        log_file = "BPP1__Test_Experiment_Results.csv"
    elif experiment_number == 2:
        WEIGHTS_OF_ITEMS = np.array(list(map(lambda i: (i**2)/2, range(1, NUMBER_OF_ITEMS+1))))
        NUMBER_OF_BINS = 50
        log_file = "BPP2_Test_Experiment_Results.csv"

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

#Result Plotting
def plot_combined_charts_with_bar_chart_and_table(experiment):
    #Load datasets
    if experiment == 1:
        results = pd.read_csv("BPP1_Test_Experiment_Results.csv")
    elif experiment == 2:
        results = pd.read_csv("BPP2_Test_Experiment_Results.csv")
    
    #Save the combined line plots to a file
    folder_name = "Experiment_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    best_fitness_per_run = results.groupby(["Experiment_Set", "Run_Number"])["Best_Fitness"].min().reset_index()

    plot_line_charts(experiment, folder_name, results)
    plot_bar_charts(experiment, folder_name, best_fitness_per_run)
    plot_tables(experiment, folder_name, best_fitness_per_run)

def plot_line_charts(experiment, folder_name, results):
    grouped = results.groupby("Experiment_Set")
    experiment_datasets = {name: group.reset_index(drop=True) for name, group in grouped}

    # Set up the 2x2 plot grid for the line plots
    figure, axis = plt.subplots(2, 2, figsize=(12, 10))
    axis = axis.flatten()

    # Iterate over each experiment set to create individual plots
    for i, (experiment_number, dataset) in enumerate(experiment_datasets.items()):
        average_data = dataset.groupby("Iteration").mean()[["Best_Fitness", "Average_Fitness", "Standard_Deviation"]]

        # Plot line charts for best fitness, average fitness, and standard deviation
        sns.lineplot(data=average_data, x=average_data.index, y="Best_Fitness", ax=axis[i], label="Best Fitness", color="darkgreen", legend=False)
        sns.lineplot(data=average_data, x=average_data.index, y="Average_Fitness", ax=axis[i], label="Average Fitness", color="darkred", legend=False)
        sns.lineplot(data=average_data, x=average_data.index, y="Standard_Deviation", ax=axis[i], label="Standard Deviation", color="blue", linestyle="--", legend=False)

        # Set titles and labels
        axis[i].set_title(f"Experiment Set {experiment_number}, p={dataset.iloc[0]['p']}, e={dataset.iloc[0]['e']}")
        axis[i].set_xlabel("Iteration")
        axis[i].set_ylabel("Fitness Value")

        # Set y-axis limit & grid lines
        axis[i].set_ylim(0, average_data.max().max() * 1.05)
        axis[i].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Create a main title for the 2x2 grid
    figure.suptitle(f"BPP{experiment} Fitnesses through Time", fontsize=16)

    # Create a single legend for the line plots at the bottom
    handles, labels = axis[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot
    save_path = os.path.join(folder_name, f"Experiment_BPP{experiment}_Metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_bar_charts(experiment, folder_name, best_fitness_per_run):
    #Plotting the bar chart for best fitness per run for each experiment set
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(data=best_fitness_per_run, x="Experiment_Set", y="Best_Fitness", hue="Run_Number", dodge=True)
    
    #Calculate the average best fitness for each Experiment_Set and add a line for each
    avg_best_fitness = best_fitness_per_run.groupby("Experiment_Set")["Best_Fitness"].mean()
    for idx, (experiment_set, avg_fitness) in enumerate(avg_best_fitness.items()):
        #Get the position of the current Experiment_Set on the x-axis
        x_position = best_fitness_per_run["Experiment_Set"].unique().tolist().index(experiment_set)
        
        #Draw a line only spanning the width of the current Experiment_Set
        plt.hlines(y=avg_fitness, xmin=x_position - 0.4, xmax=x_position + 0.4, colors='#f70ce8', linestyles='--', linewidth=1, label="Average" if idx == 0 else "")

    #Adjust the title, labels, and legend for the bar chart
    plt.title(f"BPP{experiment} Best Fitness per Run by Experiment Set")
    plt.xlabel("Experiment Set")
    plt.ylabel("Best Fitness")

    #Move the legend to below the plot
    plt.legend(title="Run Number", loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.15), fontsize=10, frameon=False)

    #Save the bar chart
    bar_chart_save_path = os.path.join(folder_name, f"Experiment_BPP{experiment}_Best_Fitness_Bar_Chart.png")
    plt.savefig(bar_chart_save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_tables(experiment, folder_name, best_fitness_per_run):
    avg_best_fitness_per_experiment = best_fitness_per_run.groupby("Experiment_Set")["Best_Fitness"].mean().reset_index()
    avg_best_fitness_per_experiment.columns = ["Experiment_Set", "Average_Best_Fitness"]

    fig, ax = plt.subplots(figsize=(6, 2))  #Adjust size as needed
    ax.axis("tight")
    ax.axis("off")
    table_data = avg_best_fitness_per_experiment.values
    table = ax.table(cellText=table_data, colLabels=avg_best_fitness_per_experiment.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    table_image_save_path = os.path.join(folder_name, f"Experiment_BPP{experiment}_Average_Best_Fitness_Table.png")
    plt.savefig(table_image_save_path, dpi=300, bbox_inches="tight")
    plt.close()

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

    #Plotting Results
    plot_combined_charts_with_bar_chart_and_table(1)
    plot_combined_charts_with_bar_chart_and_table(2)