import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data   = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data    = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=False)

class ImprovedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1  = nn.Linear(784, 256)
        self.bn1     = nn.BatchNorm1d(256)
        self.layer2  = nn.Linear(256, 128)
        self.bn2     = nn.BatchNorm1d(128)
        self.layer3  = nn.Linear(128, 64)
        self.bn3     = nn.BatchNorm1d(64)
        self.output  = nn.Linear(64, 10)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.output(x)
        return x

# train
model     = ImprovedMNIST()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("training...")
for epoch in range(10):
    model.train()
    correct = 0
    total   = 0
    for images, labels in train_loader:
        output    = model(images)
        loss      = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted  = torch.argmax(output, dim=1)
        correct   += (predicted == labels).sum().item()
        total     += labels.size(0)
    print(f"epoch {epoch+1}: accuracy={100*correct/total:.2f}%")

# save model
torch.save(model.state_dict(), 'mnist_model.pth')
print("\nmodel saved to mnist_model.pth")