import os
import glob

# Set the directory where the Python files are located
source_directory = './'

# Set the name of the output file
output_file_name = 'combined_code.txt'

# Define the divider line that will separate code from different files
divider_line = '\n\n' + '-' * 80 + '\n\n'

# Function to collect Python files recursively
def collect_python_files(path):
    python_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

# Collect all Python files in the directory and subdirectories
python_files = collect_python_files(source_directory)

# Open the output file
with open(output_file_name, 'w') as output_file:
    for file in python_files:
        # Read the content of each Python file
        with open(file, 'r') as input_file:
            file_content = input_file.read()

        # Write the file content to the output file with the divider line
        output_file.write(f'File: {os.path.relpath(file, source_directory)}\n{file_content}{divider_line}')

print(f'Combined code saved to {output_file_name}')
