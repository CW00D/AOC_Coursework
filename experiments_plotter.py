import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#results = pd.read_csv("BPP1_Experiment_Results.csv")
results = pd.read_csv("BPP2_Experiment_Results.csv")

grouped = results.groupby(['Experiment_Set'])
experiment_datasets = {}
for name, group in grouped:
    experiment_datasets[name] = group.reset_index(drop=True)

#Setting up plot area
figure, axis = plt.subplots(2, 2, figsize=(12, 10))
axis = axis.flatten()
all_fitness_values = []
for experiment_number in experiment_datasets.keys():
    #all_fitness_values.append(experiment_datasets[experiment_number]['Best_Fitness'])
    all_fitness_values.append(experiment_datasets[experiment_number]['Average_Fitness'])
y_axis_max = max([max(values) for values in all_fitness_values])*1.05

#Plot for each experiment set
for i, experiment_number in enumerate(experiment_datasets.keys()):
    dataset = experiment_datasets[experiment_number]
    
    #Plot individual runs
    #sns.lineplot(data=dataset, x='Iteration', y='Best_Fitness', hue='Run_Number', marker="o", ax=axis[i], legend=False)
    sns.lineplot(data=dataset, x='Iteration', y='Average_Fitness', hue='Run_Number', marker="o", ax=axis[i], legend=False)
    
    #Calculate and plot the average line
    #average_fitness = dataset.groupby('Iteration')['Best_Fitness'].mean()
    average_fitness = dataset.groupby('Iteration')['Average_Fitness'].mean()
    axis[i].plot(average_fitness.index, average_fitness.values, color='red', linewidth=2.5, label='Average')

    #Set consistent y-axis limits
    axis[i].set_ylim(0, y_axis_max)
    
    #Title and labels
    axis[i].set_title(f'Best Fitness Over Iterations (Experiment Set {experiment_number}, p={dataset.iloc[0]["p"]}, e={dataset.iloc[0]["e"]})')
    axis[i].set_xlabel('Iteration')
    axis[i].set_ylabel('Best Fitness')

#Adjust layout and display the plot
plt.tight_layout()

#Define the folder path
folder_name = "Experiment_Results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#Save the plot to a file
#save_path = os.path.join(folder_name, "Experiment_1_Best_Fitnesses.png")
#save_path = os.path.join(folder_name, "Experiment_2_Best_Fitnesses.png")
#save_path = os.path.join(folder_name, "Experiment_1_Average_Fitnesses.png")
save_path = os.path.join(folder_name, "Experiment_2_Average_Fitnesses.png")
plt.savefig(save_path)