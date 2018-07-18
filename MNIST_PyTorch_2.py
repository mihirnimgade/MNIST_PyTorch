import numpy as np
import torch
from mnist import MNIST
import matplotlib.pyplot as plt
import scipy.ndimage as scnd
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class TrainDatasetClass(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.training_images = torch.tensor(MNIST(root_dir).load_training()[0]).float()
        self.training_labels = torch.tensor(MNIST(root_dir).load_training()[1]).long()

    def __len__(self):
        return len(self.training_labels)

    def __getitem__(self, index):
        sample = (torch.tensor(self.training_images[index]).float(), torch.tensor(self.training_labels[index]))
        if self.transform:
            sample = (self.transform(sample[0]), sample[1])

        return sample


class TestDatasetClass(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.training_images = torch.tensor(MNIST(root_dir).load_testing()[0]).float()
        self.training_labels = torch.tensor(MNIST(root_dir).load_testing()[1]).long()

    def __len__(self):
        return len(self.training_labels)

    def __getitem__(self, index):
        sample = (torch.tensor(self.training_images[index]).float(), torch.tensor(self.training_labels[index]))
        if self.transform:
            sample = (self.transform(sample[0]), sample[1])

        return sample


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2 , num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def normalise(l):
    return l / 255.


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    input_size = 784
    hidden_size_1 = 128
    hidden_size_2 = 128
    num_classes = 10
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    # Initialisation

    TrainDataset = TrainDatasetClass(root_dir="/Users/mihirumeshnimgade/Desktop/Deep Learning", transform=normalise)
    TestDataset = TestDatasetClass(root_dir="/Users/mihirumeshnimgade/Desktop/Deep Learning")

    # Data loaders

    train_loader = torch.utils.data.DataLoader(dataset=TrainDataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=TrainDataset, batch_size=batch_size, shuffle=False)

    # Fully connected neural network with one hidden layer
    model = NeuralNet(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 784).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Accuracy of the network on the 10000 test images: {} %".format(100 * correct / total))

    torch.save(model.state_dict(), "model.ckpt")


if __name__ == '__main__':
    main()
