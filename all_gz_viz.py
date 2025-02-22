import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Parent directory containing subfolders for different label types
parent_directory = r"C:\Users\deols\OneDrive\Documents\GitHub\qml_rethinking_gen"

# List of label types
label_types_list = ["random_labels", "random_states", "real_labels"]

# Regex patterns to extract parameters from filenames
nqubits_pattern = r"nqubits_(\d+)"
training_data_pattern = r"training_data_(\d+)"

# Iterate over each label type (random_labels, random_states, real_labels)
for label_type in label_types_list:
    folder_path = os.path.join(parent_directory, label_type, "qml_error_save")
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist. Skipping.")
        continue

    print(f"\nProcessing label type: {label_type}")

    # Lists to store extracted values
    nqubits = set()
    training_data = set()

    # Get all filenames in the folder
    filenames = os.listdir(folder_path)

    # Extract unique values for nqubits and training_data
    for filename in filenames:
        if filename.endswith(".txt"):
            nqubits_match = re.search(nqubits_pattern, filename)
            training_data_match = re.search(training_data_pattern, filename)
            
            if nqubits_match:
                nqubits.add(int(nqubits_match.group(1)))
            if training_data_match:
                training_data.add(int(training_data_match.group(1)))

    # Convert sets to sorted lists
    nqubits_list = sorted(nqubits)
    training_data_list = sorted(training_data)

    # Print extracted values
    print("nqubits list:", nqubits_list)
    print("training data list:", training_data_list)

    # If 32 qubits is present, remove it
    if 32 in nqubits_list:
        nqubits_list.remove(32)

    # Colors, markers, and linestyles for plotting
    colors = ['red', 'black', 'g', 'm', 'y', 'k']
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'x']
    linestyles = ['-', '--', '-.', ':']

    # Iterate over all nqubits and training sizes to process data
    for nqubit in nqubits_list:
        train_errors = []
        test_errors = []
        gaps = []
        training_sizes = []

        for training_size in training_data_list:
            filename = f"{label_type}_nqubits_{nqubit}_training_data_{training_size}"
            print(f"Processing {filename}")

            try:
                # Load the train and test errors
                train_error = np.loadtxt(os.path.join(folder_path, f"{filename}_train_error.txt"))
                test_error = np.loadtxt(os.path.join(folder_path, f"{filename}_test_error.txt"))
            except Exception as e:
                print(f"Error loading files for {filename}: {e}")
                continue

            # Compute generalization gap
            gap = test_error - train_error

            # Append values for plotting
            train_errors.append(train_error)
            test_errors.append(test_error)
            gaps.append(gap)
            training_sizes.append(training_size)

            # Print the results
            print(f"{filename}: train_error = {train_error}, test_error = {test_error}, gap = {gap}")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(training_sizes, train_errors, label='Train Error', marker=markers[0], color=colors[0], linestyle=linestyles[0])
        plt.plot(training_sizes, test_errors, label='Test Error', marker=markers[1], color=colors[1], linestyle=linestyles[1])
        plt.plot(training_sizes, gaps, label='Generalization Gap', marker=markers[2], color=colors[2], linestyle=linestyles[2])

        # Labels and title
        plt.xlabel('Training Size')
        plt.ylabel('Error')
        plt.title(f'{label_type} - {nqubit} Qubits')
        plt.legend()
        plt.grid(True)
        plt.show()
