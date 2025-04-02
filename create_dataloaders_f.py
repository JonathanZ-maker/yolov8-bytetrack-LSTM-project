import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def create_dataloaders_f(window_size, folder_path=None, data_filename=None, batch_size=16):
    """
    Load trajectory data from a single file, normalize it, generate (X, Y) pairs, and return DataLoader.
    
    Args:
        window_size (int): Number of time steps to use as input.
        folder_path (str): Path to the folder containing .txt files.
        data_filename (str): Specific file to process (optional).
        batch_size (int): Batch size for DataLoader.

    Returns:
        dataset (TensorDataset): Dataset for testing (without DataLoader).
        scaler (MinMaxScaler): Scaler fitted on the dataset.
    """
    
    data = {}

    # ✅ Determine the file to process
    if data_filename:
        file_path = os.path.join(folder_path, data_filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Specified file not found: {file_path}")
        file_list = [data_filename]  # Process only the specified file
    else:
        file_list = sorted(f for f in os.listdir(folder_path) if f.endswith(".txt"))  # Process all files

    # ✅ Read and store data
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, sep=r'\s+')
        data[file_name] = df

    # ✅ Collect all processed data
    all_X, all_Y = [], []

    for file_name, df in data.items():
        trajectory_data = df.values  # Convert to NumPy array

        if len(trajectory_data) <= window_size:
            print(f"🚨 {file_name} has too few data points, skipping.")
            continue

        # ✅ Normalize with MinMaxScaler (-1 to 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(trajectory_data)
        normalized_data = scaler.transform(trajectory_data)

        # ✅ Generate (X, Y) pairs
        X, Y = [], []
        for i in range(len(normalized_data) - window_size):
            X.append(normalized_data[i:i + window_size])
            Y.append(normalized_data[i + window_size])

        # ✅ Convert to NumPy arrays
        X, Y = np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

        # ✅ Store results
        all_X.append(X)
        all_Y.append(Y)

    # ✅ Concatenate all processed data
    if all_X:
        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)
    else:
        raise ValueError("🚨 No valid trajectory data found!")

    # ✅ Convert to PyTorch tensors
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))

    return dataset, scaler  # ✅ Return dataset and scaler






