import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def create_dataloaders_single(window_size, folder_path=None, data_filename=None, batch_size=16):
    """
    Load trajectory data, normalize it, generate (X, Y) pairs, and return DataLoaders.
    
    Args:
        window_size (int): Number of time steps to use as input.
        folder_path (str): Path to the folder containing .txt files.
        data_filename (str): Specific file to process (optional).
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for training.
        test_dataset (TensorDataset): Dataset for testing (without DataLoader).
        scaler (MinMaxScaler): Scaler fitted on training data.
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
    all_train_X, all_train_Y = [], []
    all_test_X, all_test_Y = [], []

    for file_name, df in data.items():
        trajectory_data = df.values  # Convert to NumPy array

        # ✅ Split into training and testing (80% train, 20% test)
        split_idx = int(0.8 * len(trajectory_data))
        if split_idx <= 0:
            print(f"🚨 {file_name} has too few data points, skipping.")
            continue

        train_data = trajectory_data[:split_idx]
        test_data = trajectory_data[split_idx:]

        # ✅ Normalize with MinMaxScaler (-1 to 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        # ✅ Generate (X, Y) pairs
        train_X, train_Y = [], []
        for i in range(len(train_data) - window_size):
            train_X.append(train_data[i:i + window_size])
            train_Y.append(train_data[i + window_size])

        test_X, test_Y = [], []
        for i in range(len(test_data) - window_size):
            test_X.append(test_data[i:i + window_size])
            test_Y.append(test_data[i + window_size])

        # ✅ Convert to NumPy arrays
        train_X, train_Y = np.array(train_X, dtype=np.float32), np.array(train_Y, dtype=np.float32)
        test_X, test_Y = np.array(test_X, dtype=np.float32), np.array(test_Y, dtype=np.float32)

        # ✅ Store results
        all_train_X.append(train_X)
        all_train_Y.append(train_Y)
        all_test_X.append(test_X)
        all_test_Y.append(test_Y)

    # ✅ Concatenate all files' data
    if all_train_X:
        train_X = np.concatenate(all_train_X, axis=0)
        train_Y = np.concatenate(all_train_Y, axis=0)
        test_X = np.concatenate(all_test_X, axis=0)
        test_Y = np.concatenate(all_test_Y, axis=0)
    else:
        raise ValueError(" No valid trajectory data found!")

    # ✅ Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_Y))
    test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_Y))

    # ✅ Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_dataset, scaler  # ✅ Return scaler to reverse normalization





