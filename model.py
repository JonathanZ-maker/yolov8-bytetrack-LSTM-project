import os
import math
import random
import numpy as np
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import Timer  

# ✅ Fix all sources of randomness
def set_seed(seed=88):
    torch.manual_seed(seed)  # Fix PyTorch's random seed
    torch.cuda.manual_seed_all(seed)  # If using GPU
    np.random.seed(seed)  # Fix NumPy random seed
    random.seed(seed)  # Fix Python's built-in random seed
    torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA behavior
    torch.backends.cudnn.benchmark = False  # Disable optimizations that introduce randomness

class TrajectoryLSTM(nn.Module):
    """ The LSTM model for trajectory prediction incorporates a bidirectional LSTM, Dropout, residual connection, and MLP output layer """

    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2, dropout_rate=0.5, use_residual=True):
        super(TrajectoryLSTM, self).__init__()
        # ✅ Fix seed before initializing model layers
        set_seed(88)  # This ensures weight initialization is consistent
        # bi-LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # MLP layer as a output layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # Because it is a bidirectional LSTM, the layer dimension * 2 is hidden
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, output_size)
        )


    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
     
        out = lstm_out[:, -1, :]  # get the output of last time step

        # Prediction through MLP layer
        out = self.mlp(out)

        return out


class Model:
    """ Train and reason LSTM models """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def load_model(self, filepath):
        """ Load the trained LSTM model """
        print(f'[Model] Loading model from {filepath}')
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()  
        print('[Model] Model Loaded')

    def save_model(self, save_dir, epochs):
        """ store LSTM model """
        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.pth')
        torch.save(self.model.state_dict(), save_fname)
        print(f'[Model] Model saved at {save_fname}')

    def train(self, train_loader, epochs=10):
        """ train LSTM model """
        self.model.train()  
        timer = Timer()
        timer.start()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        timer.stop()
        self.save_model("saved_models", epochs)  

    def predict(self, data_loader):
        
        self.model.eval()  
        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions, axis=0)

