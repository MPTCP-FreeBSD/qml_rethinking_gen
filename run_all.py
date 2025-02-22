# import os
# import subprocess

# # Get the current directory of the script
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # Change the working directory to the script's directory
# os.chdir(current_directory)

# # List to store the paths of scripts to run
# scripts_to_run = []

# # Walk through the directory and find all accuracy_train.py and accuracy_test.py files
# for root, dirs, files in os.walk(current_directory):
#     for file in files:
#         if file in ["accuracy_train.py", "accuracy_test.py"]:
#             scripts_to_run.append(os.path.join(root, file))

# # Run each script in sequence
# for script in scripts_to_run:
#     script_directory = os.path.dirname(script)
#     print(f"Running {script}")
#     print(f"Changing directory to {script_directory}")
#     os.chdir(script_directory)
#     subprocess.run(["python", script], check=True)

import os
import subprocess

# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(current_directory)

# List to store the paths of scripts to run
scripts_to_run = []

# List of folder names to include
include = ['8']

# Walk through the directory and find all accuracy_train.py and accuracy_test.py files
for root, dirs, files in os.walk(current_directory):
    for file in files:
        if file in ["accuracy_train.py", "accuracy_test.py"]:
            print("root:",root)
            print("file:",file)
            if any(incl in root for incl in include):
                scripts_to_run.append(os.path.join(root, file))

# Function to extract the folder name and qubit number for sorting
def extract_sorting_key(path):
    parts = path.split("\\")
    
    # Extract folder name (e.g., random_labels, random_states, real_labels)
    folder_name = parts[-3]  # Folder name should be the third last in the path
    
    # Extract the number of qubits (assuming the format includes <number>_qubits)
    for part in parts:
        if "qubits" in part:
            qubit_number = int(part.split("_")[0])  # Get the qubit number
            return (folder_name, qubit_number)
    
    return (folder_name, float('inf'))  # Default if no qubit number is found

# Sort the list first by folder name alphabetically, then by qubit number
scripts_to_run.sort(key=extract_sorting_key)
print(scripts_to_run)
print("&&"*20)

# Run each script in sequence
for script in scripts_to_run:
    script_directory = os.path.dirname(script)
    print(f"Running {script}")
    print(f"Changing directory to {script_directory}")
    os.chdir(script_directory)
    subprocess.run(["python", script], check=True)