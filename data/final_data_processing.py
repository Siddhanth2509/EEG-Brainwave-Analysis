import os
import pandas as pd
import scipy.io as sio
import numpy as np
import re

# ------------------------------------------------------------
# Configuration Section
# ------------------------------------------------------------
# Directory containing the raw EEG .mat files
DATA_DIR = 'raw_data_folder'  # Example: 'Clean_Cliped'

# Name of the EEG variable inside each .mat file
# If None, the script will automatically detect it from the first file
MAT_VARIABLE_NAME = None

# ------------------------------------------------------------
# Main Script
# ------------------------------------------------------------
print("Starting EEG data processing...")

# This list will hold the processed DataFrames for each file
all_data_list = []

# ------------------------------------------------------------
# Step 1: Collect all .mat files from the specified directory
# ------------------------------------------------------------
try:
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
    if not file_list:
        print(f"Error: No .mat files found in '{DATA_DIR}'. Please check the path.")
        exit()
except FileNotFoundError:
    print(f"Error: The directory '{DATA_DIR}' was not found. Please check the path.")
    exit()

print(f"Found {len(file_list)} files to process.")

# ------------------------------------------------------------
# Step 2: Process each EEG file
# ------------------------------------------------------------
for i, filename in enumerate(file_list):
    print(f"Processing file {i+1}/{len(file_list)}: {filename}")

    # --------------------------------------------------------
    # Extract subject ID and emotion label from the filename
    # The pattern assumes filenames like 'sub01t1H.mat' where:
    #   H = Happy, S = Sad, F = Fear
    # --------------------------------------------------------
    match = re.search(r'sub(\d+)t\d+([HSF])', filename)
    if not match:
        print(f"  - Warning: Could not parse subject and emotion from '{filename}'. Skipping.")
        continue

    subject_id = int(match.group(1))
    emotion_code = match.group(2)

    # Map emotion code to descriptive label
    emotion_map = {'H': 'Happy', 'S': 'Sad', 'F': 'Fear'}
    emotion = emotion_map.get(emotion_code, 'Unknown')

    # --------------------------------------------------------
    # Load the .mat file
    # --------------------------------------------------------
    file_path = os.path.join(DATA_DIR, filename)
    try:
        mat_contents = sio.loadmat(file_path)
    except Exception as e:
        print(f"  - Error loading file {filename}: {e}. Skipping.")
        continue

    # --------------------------------------------------------
    # Identify the EEG data variable
    # If MAT_VARIABLE_NAME is None, auto-detect it based on size
    # --------------------------------------------------------
    if MAT_VARIABLE_NAME is None:
        best_key = None
        max_size = 0
        for key, value in mat_contents.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                if value.size > max_size:
                    max_size = value.size
                    best_key = key

        if best_key:
            MAT_VARIABLE_NAME = best_key
            print(f"Auto-detected EEG data variable as: '{MAT_VARIABLE_NAME}'")
        else:
            print(f"Error: Could not automatically find the data variable in {filename}.")
            print("Please inspect the .mat file and set 'MAT_VARIABLE_NAME' manually.")
            print("Variables found:", list(mat_contents.keys()))
            exit()

    if MAT_VARIABLE_NAME not in mat_contents:
        print(f"  - Error: Variable '{MAT_VARIABLE_NAME}' not found in {filename}. Skipping.")
        print(f"    Available variables: {list(mat_contents.keys())}")
        continue

    eeg_data = mat_contents[MAT_VARIABLE_NAME]

    # --------------------------------------------------------
    # Ensure the EEG data format is consistent
    # The dataset is expected to have 32 channels (columns)
    # --------------------------------------------------------
    if eeg_data.shape[0] == 32:
        eeg_data = eeg_data.T  # Transpose if channels are in rows

    if eeg_data.shape[1] != 32:
        print(f"  - Warning: Expected 32 channels, but found {eeg_data.shape[1]} in {filename}. Skipping.")
        continue

    # --------------------------------------------------------
    # Create a DataFrame for the current subject
    # Each column represents one EEG channel
    # --------------------------------------------------------
    temp_df = pd.DataFrame(eeg_data, columns=[f'ch_{i+1}' for i in range(32)])
    temp_df['subject_id'] = subject_id
    temp_df['emotion'] = emotion

    all_data_list.append(temp_df)

# ------------------------------------------------------------
# Step 3: Combine all subject data and save as a CSV file
# ------------------------------------------------------------
if all_data_list:
    print("Combining all dataframes into a single dataset...")

    final_df = pd.concat(all_data_list, ignore_index=True)

    # Reorder columns so identifiers appear first
    cols = ['subject_id', 'emotion'] + [f'ch_{i+1}' for i in range(32)]
    final_df = final_df[cols]

    output_filename = 'eeg_emotion_dataset.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"Success! Combined data saved to '{output_filename}'.")
    print("Final dataset shape:", final_df.shape)
else:
    print("No data was processed. The final CSV file was not created.")
