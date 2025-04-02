import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from create_dataloaders_single import create_dataloaders_single
from model import TrajectoryLSTM

# ✅ Define dataset folder path
folder_path = "cam_test_seperate/s4/front"  # Update this with your actual dataset path
#data_filename = "trajectory_MOT16-02_54.txt"  # Update this with the actual file name

# ✅ Set the best parameters (update these based on your training results)
BEST_WINDOW_SIZE = 8  # Update based on best result
BEST_HIDDEN_SIZE = 128  # Update based on best result
BEST_DROPOUT = 0.3  # Update based on best result

# ✅ Construct Model Path Dynamically
current_dir = os.getcwd()  # Get current working directory
model_dir = os.path.join(current_dir, "models")  # Path to model folder
model_filename = "best_sm_model.pth"  # Model file name
MODEL_PATH = os.path.join(model_dir, model_filename)  # Full path to model file

# ✅ Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

print(f"✅ Loading Model from: {MODEL_PATH}")

# ✅ Load Test Data
_, test_dataset, scaler = create_dataloaders_single(BEST_WINDOW_SIZE, folder_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Trained Model
model = TrajectoryLSTM(input_size=2, hidden_size=BEST_HIDDEN_SIZE, dropout_rate=BEST_DROPOUT).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)  # SAFE LOAD
model.load_state_dict(state_dict)
model.eval()

# ✅ Get Test Data
X_test, Y_test = test_dataset[:]
X_test = X_test.to(device)

# ✅ Make Predictions
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()

# ✅ Reverse Normalization
y_true = scaler.inverse_transform(Y_test)
y_pred = scaler.inverse_transform(y_pred)

# ✅ Compute MSE
mse = mean_squared_error(y_true, y_pred)
print(f"\n🏆 Test Set MSE: {mse:.4f}")

# ✅ Plot Predictions vs. Ground Truth
plt.figure(figsize=(8, 6))
plt.scatter(y_true[:, 0], y_true[:, 1], label="True Trajectory", color="red", alpha=0.5)
plt.scatter(y_pred[:, 0], y_pred[:, 1], label="Predicted Trajectory", color="blue", alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Model Prediction vs. Ground Truth")
plt.legend()
plt.show()
plt.savefig("testset_comparison.png")
print("✅ Test Set Comparison Saved: testset_comparison.png")




