"""import torch
import torch.nn as nn

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x
        
nn_module = neural_network()
print(nn_module)

print("\nhidden layer weights shape:", nn_module.hidden.weight.shape)
print("hidden layer bias shape:", nn_module.hidden.bias.shape)
print("output layer weights shape:", nn_module.output.weight.shape)

x = torch.tensor([0.0, 0.0])
output = nn_module(x)
print("input:", x)
print("output:", output)
print("output shape:", output.shape)

import torch.optim as optim

# loss function and optimizer
criterion = nn.BCELoss()          # Binary Cross Entropy - for 0/1 classification
optimizer = optim.SGD(nn_module.parameters(), lr=0.1)  # Stochastic Gradient Descent

# test loss calculation
target = torch.tensor([0.0])      # correct answer for input [0,0] is 0
loss   = criterion(output, target)
print("loss:", loss)

# XOR dataset
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# training loop
for i in range(10000):
    total_loss = 0

    for j in range(len(X)):
        # forward pass
        output = nn_module(X[j])
        loss   = criterion(output, y[j])
        total_loss += loss.item()

        # backward pass
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # calculate new gradients
        optimizer.step()        # update weights

    if (i+1) % 1000 == 0:
        print(f"iteration {i+1}: loss={total_loss/len(X):.6f}")

print("\n--- Predictions ---")
with torch.no_grad():
    for j in range(len(X)):
        output    = nn_module(X[j])
        predicted = 1 if output.item() > 0.5 else 0
        print(f"input: {X[j].tolist()} → output: {output.item():.4f} → predicted: {predicted} → actual: {int(y[j].item())}")"""
        
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# download and load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),                  # convert image to tensor
    transforms.Normalize((0.5,), (0.5,))    # normalize pixels to -1 to 1
])

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=False)

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))
print("Input shape:", train_data[0][0].shape)
        
class MNISTNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()           # 28×28 → 784
        self.hidden  = nn.Linear(784, 128)    # 784 inputs, 128 hidden neurons
        self.output  = nn.Linear(128, 10)     # 128 inputs, 10 outputs (digits 0-9)

    def forward(self, x):
        x = self.flatten(x)                   # flatten image to vector
        x = torch.relu(self.hidden(x))        # ReLU for hidden layer
        x = self.output(x)                    # no activation - CrossEntropyLoss handles it
        return x

model     = MNISTNetwork()
criterion = nn.CrossEntropyLoss()             # combines softmax + cross entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model)
def train(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        correct    = 0
        total      = 0

        for images, labels in train_loader:
            # forward pass
            output = model(images)
            loss   = criterion(output, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track accuracy
            total_loss += loss.item()
            predicted   = torch.argmax(output, dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

        accuracy = 100 * correct / total
        print(f"epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, accuracy={accuracy:.2f}%")

train(model, train_loader, criterion, optimizer, epochs=5)

def evaluate(model, test_loader):
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in test_loader:
            output    = model(images)
            predicted = torch.argmax(output, dim=1)
            correct  += (predicted == labels).sum().item()
            total    += labels.size(0)

    accuracy = 100 * correct / total
    print(f"test accuracy: {accuracy:.2f}%")

evaluate(model, test_loader)
        