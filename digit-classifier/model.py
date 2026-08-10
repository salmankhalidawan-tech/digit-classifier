import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for digit recognition.
    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel
    - Pool1: 2x2 max pooling
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - Pool2: 2x2 max pooling
    - FC1: 64 * 7 * 7 -> 128
    - FC2: 128 -> 10 (Output logits)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # 28x28 -> pool -> 14x14 -> pool -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Layer 1
        x = self.pool(F.relu(self.conv1(x)))
        # Layer 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
