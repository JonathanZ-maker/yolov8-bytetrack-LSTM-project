import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from joblib import dump
from itertools import product
from sklearn.metrics import mean_squared_error
from create_dataloaders_single import create_dataloaders_single
from model import TrajectoryLSTM

# Define dataset folder path
folder_path = "cam_test/mot-02"  # Update this with your actual dataset path

# Fix all sources of randomness
def set_seed(seed=88):
    torch.manual_seed(seed)  # Fix PyTorch's random seed
    torch.cuda.manual_seed_all(seed)  # If using GPU
    np.random.seed(seed)  # Fix NumPy random seed
    random.seed(seed)  # Fix Python's built-in random seed
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA behavior
    torch.backends.cudnn.benchmark = False  # Disable optimizations that introduce randomness

set_seed(88)  # Call this at the very start

# Define Grid Search Parameters
window_sizes = [4, 8, 16]  # Test different sequence lengths
learning_rates = [0.001, 0.0005, 0.0001]
hidden_sizes = [64, 128, 256]
dropout_rates = [0.2, 0.3, 0.4]

# Track best model
best_mse = float("inf")
best_params = None
results = []

#folder_path="cam_test/s3"
# Grid Search Loop
for window_size, lr, hidden_size, dropout_rate in product(
    window_sizes, learning_rates, hidden_sizes, dropout_rates
):
    print(f"\n🚀 Testing: window_size={window_size}, lr={lr}, hidden_size={hidden_size}, dropout={dropout_rate}")
    # Step 1: Create DataLoaders with new window_size
    train_loader, test_dataset, scaler = create_dataloaders_single(window_size, folder_path)
    
    # Step 2: Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryLSTM(input_size=2, hidden_size=hidden_size, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Step 3: Train Model
    num_epochs = 40
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"📉 Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Step 4: Evaluate on Test Set
    model.eval()
    X_test, Y_test = test_dataset[:]
    X_test = X_test.to(device)

    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    # Reverse Normalization
    y_true = scaler.inverse_transform(Y_test)
    y_pred = scaler.inverse_transform(y_pred)

    # Compute MSE
    mse = mean_squared_error(y_true, y_pred)
    print(f"🎯 MSE for (window_size={window_size}, lr={lr}, hidden={hidden_size}, dropout={dropout_rate}): {mse:.4f}")

    # Track best parameters
    results.append((window_size, lr, hidden_size, dropout_rate, mse))
    if mse < best_mse:
        best_mse = mse
        best_params = (window_size, lr, hidden_size, dropout_rate)

# Display Best Hyperparameters
print(f"\n🏆 Best Parameters: window_size={best_params[0]}, lr={best_params[1]}, hidden_size={best_params[2]}, dropout={best_params[3]}")
print(f" Best Test MSE: {best_mse:.4f}")

# Save Grid Search Results
results.sort(key=lambda x: x[-1])  # Sort by MSE
with open("grid_search_results.txt", "w") as f:
    for res in results:
        f.write(f"window_size={res[0]}, lr={res[1]}, hidden_size={res[2]}, dropout={res[3]}, MSE={res[4]:.4f}\n")
print("Grid Search Completed and Results Saved.")

#--------------------------------------------------
#Get results and save the best model
#  Step 1: Train with Best Parameters
print("\n🚀 Training Final Model with Best Parameters...")

best_window_size, best_lr, best_hidden_size, best_dropout = best_params

# Create DataLoaders with best window_size
train_loader, test_dataset, scaler = create_dataloaders_single(best_window_size,folder_path)
# Initialize Model with Best Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model = TrajectoryLSTM(input_size=2, hidden_size=best_hidden_size, dropout_rate=best_dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_lr)

# Train the Final Model
num_epochs = 40
final_train_losses = []

for epoch in range(num_epochs):
    final_model.train()
    running_loss = 0.0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    final_train_losses.append(avg_loss)
    print(f"Final Model Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the Trained Model
model_save_path = "best_trajectory_model.pth"
torch.save(final_model.state_dict(), model_save_path)
print(f"Final Model Saved: {model_save_path}")

# Step 2: Evaluate Final Model
final_model.eval()
X_test, Y_test = test_dataset[:]
X_test = X_test.to(device)

with torch.no_grad():
    y_pred = final_model(X_test).cpu().numpy()

# Reverse Normalization
y_true = scaler.inverse_transform(Y_test)
y_pred = scaler.inverse_transform(y_pred)

# Compute Final MSE
final_mse = mean_squared_error(y_true, y_pred)
print(f"\n Final Model Test MSE: {final_mse:.4f}")

# Step 3: Plot Test Set Predictions vs. Ground Truth
plt.figure(figsize=(8, 6))
plt.scatter(y_true[:, 0], y_true[:, 1], label="True Trajectory", color="red", alpha=0.5)
plt.scatter(y_pred[:, 0], y_pred[:, 1], label="Predicted Trajectory", color="blue", alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Final Model Prediction vs. Ground Truth")
plt.legend()

# Show and Save the Figure
plt.show()
plt.savefig("final_testset_comparison.png")
print("Final Test Set Comparison Saved: final_testset_comparison.png")


