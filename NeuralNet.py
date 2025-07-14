import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

		self.fc1 = nn.Linear(64 * 8 * 8, 128)
		self.fc2 = nn.Linear(128, 64)
		self.fc3 = nn.Linear(64, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		
		x = x.view(-1, 64 * 8 * 8)
		
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)