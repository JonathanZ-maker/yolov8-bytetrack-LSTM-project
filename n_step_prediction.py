import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TrajectoryLSTM
from create_dataloaders_f import create_dataloaders_f
from sklearn.metrics import mean_squared_error

# --- Configuration ---
folder_path = "cam_test/s1-4"
data_filename = "s2_1.txt"
BEST_WINDOW_SIZE = 8
BEST_HIDDEN_SIZE = 128
BEST_DROPOUT = 0.3
N_STEPS = 10 # 🔁 How many steps to predict

# --- Load Model ---
model_dir = os.path.join(os.getcwd(), "models")
model_filename = "best_sm_model.pth"
MODEL_PATH = os.path.join(model_dir, model_filename)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

print(f"✅ Loading model from: {MODEL_PATH}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrajectoryLSTM(input_size=2, hidden_size=BEST_HIDDEN_SIZE, dropout_rate=BEST_DROPOUT).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --- Load Test Data ---
dataset, scaler = create_dataloaders_f(BEST_WINDOW_SIZE, folder_path, data_filename)
X_test, Y_test = dataset[:]  # X_test: (N, window, 2), Y_test: (N, 2)

# --- Predict for Entire Test Set ---
predictions = []
true_values = []

with torch.no_grad():
    for i in range(len(X_test)):
        current_seq = X_test[i].unsqueeze(0).to(device)  # (1, window_size, 2)
        pred_seq = []

        for _ in range(N_STEPS):
            next_point = model(current_seq)  # (1, 2)
            pred_np = next_point.squeeze(0).cpu().numpy()
            pred_seq.append(pred_np)

            # Update sequence with predicted step
            pred_tensor = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 2)
            current_seq = torch.cat([current_seq[:, 1:, :], pred_tensor], dim=1)

        predictions.append(np.array(pred_seq))
        true_values.append(Y_test[i].cpu().numpy())

# --- Inverse Normalize ---
pred_np = scaler.inverse_transform(np.vstack(predictions))
true_np = scaler.inverse_transform(np.vstack(true_values))

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.plot(true_np[:, 0], true_np[:, 1], 'go-', label="True Trajectory")
plt.plot(pred_np[:, 0], pred_np[:, 1], 'bo--', label="Predicted Trajectory")

plt.title(f"{N_STEPS}-Step Prediction for Entire Test Set")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("n_step_prediction_all.png")
plt.show()
print("✅ Saved: n_step_prediction_all.png")

# --- Compute Step-wise MSE ---
step_indices = [0, 4, 9, 14] 
step_labels = ["1-step", "5-step", "10-step", "15-step"]

mse_results = {}

for step_idx, label in zip(step_indices, step_labels):
    step_preds = []
    step_truths = []

    for i in range(len(predictions)):
        if step_idx < len(predictions[i]):
            step_preds.append(predictions[i][step_idx])
            step_truths.append(Y_test[i].cpu().numpy()) 

    step_preds = scaler.inverse_transform(np.array(step_preds))
    step_truths = scaler.inverse_transform(np.array(step_truths))

    mse = mean_squared_error(step_truths, step_preds)
    mse_results[label] = mse
    print(f"MSE for {label} prediction: {mse:.4f}")
