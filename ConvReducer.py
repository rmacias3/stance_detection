import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvReducer(nn.Module):
    def __init__(self):
        super(ConvReducer, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def forward(self, x):
        conv1 = self.bn1(self.relu(self.conv1(x)))
        conv2 = self.bn2(self.relu(self.conv2(conv1)))
        conv3 = self.bn3(self.relu(self.conv3(conv2))).view(-1, 8 * 8 * 16)
        fc1 = self.selu(self.fc1(conv3))
        features = self.fc2(fc1)
        return features