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

device = torch.device("cpu")

def one_hot(number_array, length=10):
	x = torch.zeros([len(number_array), length], dtype=torch.uint8)
	one_hot_array = torch.zeros(length)
	for i in range(0, len(number_array)):
		one_hot_array[number_array[i]] = 1
		x[i] = one_hot_array
		one_hot_array = torch.zeros(length)
	return x

def normalise(l):
    return l / 255.

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


# mndata = MNIST("/Users/mihirumeshnimgade/Desktop/Deep Learning")
# training_images, training_labels = mndata.load_training()
#
# training_images = torch.Tensor(training_images)
#
# training_labels = torch.Tensor(training_labels)

# training_labels = one_hot(training_labels.long())


batch_size = 128
input_dim = 784
hidden_1, hidden_2 = 16, 16
output_dim = 10
num_epochs = 10

# Initialisation

TrainDataset = TrainDatasetClass(root_dir="/Users/mihirumeshnimgade/Desktop/Deep Learning", transform=normalise)

# Data loaders

train_loader = torch.utils.data.DataLoader(dataset=TrainDataset, batch_size=batch_size, shuffle=True)


class MNIST_DNN(nn.Module):
    def __init__(self, input_dim, hidden_1, hidden_2, output_dim):
        super(MNIST_DNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.linear3 = nn.Linear(hidden_2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1_relu = self.relu(self.linear1(x))
        h2_relu = self.relu(self.linear2(h1_relu))
        y_pred = self.linear3(h2_relu)
        return y_pred

model = MNIST_DNN(input_dim, hidden_1, hidden_2, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(device)
        labels = labels.long().to(device)

        outputs = model(images)



        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(epoch, loss.item())


with torch.no_grad():
    number = scnd.imread("/Users/mihirumeshnimgade/Desktop/number7.png")
    flattened_number = number.reshape((1,784))
    flattened_number = flattened_number.astype(float)
    flattened_number = np.interp(flattened_number, (flattened_number.min(), flattened_number.max()), (1, 0))
    flattened_number = torch.from_numpy(flattened_number)
    y_pred = model(flattened_number.float())
    print(y_pred)
