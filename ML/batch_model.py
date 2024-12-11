import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from adaptive_window import AdaptiveWindow

class Batch_MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(Batch_MLPModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.softmax(x)

class Batch_LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Batch_LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        return self.softmax(x)


def train_batch_model(model, adaptive_window, optimizer, criterion):
    for batch_data in adaptive_window.batches:
        inputs = torch.tensor(batch_data[:, :-1], dtype=torch.float32)
        labels = torch.tensor(batch_data[:, -1], dtype=torch.long)

     #   print("Input shape:", inputs.shape)  # 打印输入形状，确保为 [batch_size, num_features]
      #  print("Label shape:", labels.shape)  # 打印标签形状，确保为 [batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    #print("Model updated with batch data.")


def predict_batch_model(model, inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted

def save_batch(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_batch(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
