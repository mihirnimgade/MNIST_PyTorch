import matplotlib.pyplot as plt
import scipy.ndimage as scnd
import torch
# from MNIST_Stuff import NeuralNet
# from MNIST_Stuff import normalise
import numpy as np
import torch.nn as nn

#number = scnd.imread("/Users/mihirumeshnimgade/Desktop/number7.png")
#
#number = torch.from_numpy(number)
#print(number.size())
#print(type(number))

#model = load_state_dict(torch.load("/Users/mihirumeshnimgade/Desktop/model.ckpt"))
#model.eval()
#print(type(model))

device = torch.device("cpu")


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

model = NeuralNet(784, 128, 128, 10)
model.load_state_dict(torch.load("model.ckpt"))



with torch.no_grad():
	number = scnd.imread("/Users/mihirumeshnimgade/Desktop/mnistnumber.png")
#	plt.imshow(number)
#	plt.show()
	
	flattened_number = number.reshape((1,784))
	flattened_number = flattened_number.astype(float)
#	flattened_number = np.interp(flattened_number, (flattened_number.min(), flattened_number.max()), (1, 0))
	flattened_number = normalise(torch.from_numpy(flattened_number))
	
	y_pred = model(flattened_number.float())
#	print(y_pred)
	print(np.argmax(y_pred))