import csv
import os

# Define the log file path
log_file = 'aco_experiment_log.csv'

# Initialize log file with headers (run once at the start)
def initialize_log():
    if not os.path.exists(log_file):  # Avoid overwriting if it already exists
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Trial', 'Problem_Type', 'p', 'e', 'Best_Fitness', 'Iteration'])

# Log results for each trial
def log_results(trial, problem_type, p, e, best_fitness, iteration):
    with open(log_file, mode='a', newline='') as file:  # Append mode
        writer = csv.writer(file)
        writer.writerow([trial, problem_type, p, e, best_fitness, iteration])

# Example of running trials and logging
def run_trials():
    initialize_log()
    
    for trial in range(1, 6):  # Run 5 trials
        best_fitness = run_aco(k=500, b=10, p=100, e=0.9, iterations=10000)
        log_results(trial, 'BPP1', 100, 0.9, best_fitness, 10000)
        
        best_fitness = run_aco(k=500, b=50, p=10, e=0.6, iterations=10000)
        log_results(trial, 'BPP2', 10, 0.6, best_fitness, 10000)
