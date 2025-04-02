import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from create_dataloaders_f import create_dataloaders_f
from model import TrajectoryLSTM

# ✅ Define dataset file (Single File Test)
data_filename = "trajectory_MOT16-02_61.txt"  # Update this with your actual file name
folder_path = "cam_test/mot-02_test"  # Folder containing the file

# ✅ Define two models for comparison
models_config = [
    {"window_size": 4, "hidden_size": 256, "dropout": 0.3, "model_name": "best_mot02_model.pth", "color": "blue"},
    {"window_size": 8, "hidden_size": 128, "dropout": 0.3, "model_name": "best_sm_model.pth", "color": "green"},
]

# ✅ Construct Model Paths
current_dir = os.getcwd()
model_dir = os.path.join(current_dir, "models")

# ✅ Ensure model files exist
for config in models_config:
    config["model_path"] = os.path.join(model_dir, config["model_name"])
    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"❌ Model file not found: {config['model_path']}")

# ✅ Generate Test Datasets for Different Window Sizes
test_datasets = {}
scalers = {}

for config in models_config:
    window_size = config["window_size"]
    print(f"📌 Preparing test dataset for window size: {window_size}")

    test_dataset, scaler = create_dataloaders_f(
        window_size=window_size, 
        folder_path=folder_path, 
        data_filename=data_filename
    )

    test_datasets[window_size] = test_dataset
    scalers[window_size] = scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Store results
mse_results = {}
predictions = {}  # Store predictions for plotting

# ✅ Loop through each model for testing
for config in models_config:
    window_size = config["window_size"]
    model_name = config["model_name"]
    print(f"\n🔍 Evaluating Model: {model_name} (Window Size: {window_size})")

    # ✅ Load the corresponding test dataset
    test_dataset = test_datasets[window_size]
    scaler = scalers[window_size]

    # ✅ Get Test Data
    X_test, Y_test = test_dataset[:]
    X_test = X_test.to(device)

    # ✅ Load Model
    model = TrajectoryLSTM(
        input_size=2, 
        hidden_size=config["hidden_size"], 
        dropout_rate=config["dropout"]
    ).to(device)

    state_dict = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ✅ Make Predictions
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    # ✅ Reverse Normalization
    y_true = scaler.inverse_transform(Y_test)
    y_pred = scaler.inverse_transform(y_pred)

    # ✅ Store predictions for plotting
    predictions[model_name] = y_pred

    # ✅ Compute MSE
    mse = mean_squared_error(y_true, y_pred)
    mse_results[model_name] = mse
    print(f"🏆 Model {model_name} - Test Set MSE: {mse:.4f}")

# ✅ Plot All Predictions Together
plt.figure(figsize=(10, 7))

# ✅ Plot Ground Truth
plt.plot(y_true[:, 0], y_true[:, 1], label="True Trajectory", color="red", linestyle="-", linewidth=2)

# ✅ Plot Model Predictions
for config in models_config:
    model_name = config["model_name"]
    plt.plot(predictions[model_name][:, 0], predictions[model_name][:, 1], 
             label=f"Predicted - {model_name}", color=config["color"], linestyle="--", linewidth=2)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Model Comparison: Prediction vs. Ground Truth")
plt.legend()
plt.savefig("testset_comparison_multi_model.png")
plt.show()

print("✅ Test Set Comparison Saved: testset_comparison_multi_model.png")

# ✅ Print All MSE Scores
print("\n📊 Model Comparison Results:")
for model_name, mse in mse_results.items():
    print(f"🔹 {model_name}: MSE = {mse:.4f}")







