import os
import matplotlib.pyplot as plt
import numpy as np
import csv

output_folder = "outputs"

output_files = os.listdir(output_folder)

required_output_files = [f for f in output_files if f[:12] == "SVRT Problem"]

required_output_files.sort()

problem_results = {}

column_headers = ["learning_rate","training_data_size","architecture","fold_index","accuracy_after_training","accuracy_of_best_weights"]

for problem_file in required_output_files:
    file_path = os.path.join(output_folder, problem_file)
    with open(file_path) as f:
        results_lines = f.read().split("\n")
        results = []
        for results_line in results_lines:
            if(results_line == ""):
                continue
            results.append(dict(zip(column_headers, results_line.split(","))))    
        problem_number = int(problem_file.split(" ")[2].split("_")[0]) #SVRT Problem 1_training_outputs
        problem_results[problem_number] = results


problem_numbers = list(range(1,24))

model_name = "vgg16"
training_data_size = "2000"


accuracies = []

for problem_number in problem_numbers:
    all_problem_results = problem_results[problem_number]
    problem_results_for_experiment = [r for r in all_problem_results if r["architecture"]==model_name and r["training_data_size"]==training_data_size]

    problem_accuracies = [float(r["accuracy_of_best_weights"]) for r in problem_results_for_experiment]
    accuracies.append(problem_accuracies)

accuracies = np.array(accuracies)
means = accuracies.mean(1)
mins = accuracies.min(1)
maxes = accuracies.max(1)
stds = accuracies.std(1)

plt.errorbar(problem_numbers, means, stds, fmt='ok', lw=3)
plt.errorbar(problem_numbers, means, [means - mins, maxes - means],
             fmt='.k', ecolor='gray', lw=1)

# x = np.array([1, 2, 3, 4, 5])
# y = np.power(x, 2) # Effectively y = x**2
# e = np.array([1.5, 2.6, 3.7, 4.6, 5.5])

# plt.errorbar(x, y, e, linestyle='None', marker='^')

plt.show()

means_accuracies = zip(problem_numbers,means)

for m in means_accuracies:
    print(m)
