# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:03:41 2023

@author: florianma
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
print(torch.cuda.is_available())


pth = "C:/Users/florianma/OneDrive - Institutt for Energiteknikk/Documents/data/VTK_Legacy_NEW/"
# Load the input and output data
input_data = np.load(pth+'data_on_grid_s.npy')
output_data = np.load(pth+'data_on_grid.npy')

# Convert the data to PyTorch tensors
input_tensor = torch.from_numpy(input_data.T).float()
output_tensor = torch.from_numpy(output_data.T).float()

# Define the neural network architecture


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_tensor.shape[1])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Create an instance of the neural network
model = NeuralNetwork()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the number of training epochs
num_epochs = 1000

# Train the neural network
for epoch in range(num_epochs):
    # Shuffle the data for each epoch
    indices = torch.randperm(len(input_data.T))
    shuffled_input = input_tensor[indices]
    shuffled_output = output_tensor[indices]

    # Process each batch
    num_batches = 30
    batch_size = 100
    for batch in range(num_batches):
        # Extract the batch data
        start = batch * batch_size
        end = start + batch_size
        batch_input = shuffled_input[start:end]
        batch_output = shuffled_output[start:end]

        # Forward pass
        outputs = model(batch_input)
        loss = criterion(outputs, batch_output)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss every 100 epochs
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
