import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_combined_charts(experiment):
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

    #Iterate over each experiment set to create individual plots
    for i, (experiment_number, dataset) in enumerate(experiment_datasets.items()):
        average_data = dataset.groupby("Iteration").mean()[["Best_Fitness", "Average_Fitness", "Standard_Deviation"]]
        
        #Plot lines
        sns.lineplot(data=average_data, x=average_data.index, y="Best_Fitness", ax=axis[i], label="Best Fitness", color="darkgreen", legend=False)
        sns.lineplot(data=average_data, x=average_data.index, y="Average_Fitness", ax=axis[i], label="Average Fitness", color="darkred", legend=False)
        sns.lineplot(data=average_data, x=average_data.index, y="Standard_Deviation", ax=axis[i], label="Standard Deviation", color="blue", linestyle="--", legend=False)
        
        #Set titles and labels
        axis[i].set_title(f"(Experiment Set {experiment_number}, p={dataset.iloc[0]['p']}, e={dataset.iloc[0]['e']})")
        axis[i].set_xlabel("Iteration")
        axis[i].set_ylabel("Fitness Value")

        #Set y-axis limit & grid lines
        axis[i].set_ylim(0, average_data.max().max() * 1.05)
        axis[i].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    #Create a single legend from the last plot
    handles, labels = axis[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    #Save the plot to a file
    folder_name = "Experiment_Results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    save_path = os.path.join(folder_name, f"Experiment_BBP{experiment}_Metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

#Generate and save combined plots for both experiments
for i in range(1, 3):
    plot_combined_charts(i)
