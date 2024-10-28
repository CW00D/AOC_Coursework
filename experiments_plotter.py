import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_charts(experiment, metric_to_plot):
    #Load the appropriate dataset based on the experiment parameter
    if experiment == 1:
        results = pd.read_csv("BPP1_Experiment_Results.csv")
    elif experiment == 2:
        results = pd.read_csv("BPP2_Experiment_Results.csv")

    #Group by "Experiment_Set" and create individual datasets
    grouped = results.groupby("Experiment_Set")
    experiment_datasets = {name: group.reset_index(drop=True) for name, group in grouped}

    #Set up the 2x2 plot grid
    figure, axis = plt.subplots(2, 2, figsize=(12, 10))
    axis = axis.flatten()

    #Determine a common y-axis limit across all experiment sets for consistency
    all_fitness_values = [experiment_datasets[exp_num][metric_to_plot] for exp_num in experiment_datasets.keys()]
    y_axis_max = max([max(values) for values in all_fitness_values]) * 1.05

    #Iterate over each experiment set to create individual plots
    for i, experiment_number in enumerate(experiment_datasets.keys()):
        dataset = experiment_datasets[experiment_number]

        #Plot individual runs on the primary y-axis
        sns.lineplot(data=dataset, x="Iteration", y=metric_to_plot, hue="Run_Number", marker="o", ax=axis[i], legend="full")
        
        #Calculate and plot the average line on the primary y-axis
        average_fitness = dataset.groupby("Iteration")[metric_to_plot].mean()
        axis[i].plot(average_fitness.index, average_fitness.values, color="red", linewidth=2.5, label="Average")
        
        #Set consistent y-axis limits for the primary axis
        axis[i].set_ylim(0, y_axis_max)

        #Titles and labels for the primary y-axis
        axis[i].set_title(f"{metric_to_plot} Over Iterations\n(Experiment Set {experiment_number}, p={dataset.iloc[0]['p']}, e={dataset.iloc[0]['e']})")
        axis[i].set_xlabel("Iteration")
        axis[i].set_ylabel(f"{metric_to_plot}")

    #Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    #Define the output folder path and create it if it doesn"t exist
    folder_name = "Experiment_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #Save the plot to a file with the metric in the filename
    save_path = os.path.join(folder_name, f"Experiment_{experiment}_{metric_to_plot}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

#Generate and save plots for both Best_Fitness and Average_Fitness
for i in range(1, 3):
    plot_charts(i, "Best_Fitness")
    plot_charts(i, "Average_Fitness")
    plot_charts(i, "Standard_Deviation")
