import torch
import torch.nn as nn
from torchvision import datasets, transforms

# define same architecture
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

# step 1 - create empty model
model = ImprovedMNIST()

# step 2 - load saved weights
model.load_state_dict(torch.load('mnist_model.pth'))
print("model loaded successfully")

# step 3 - set to evaluation mode
model.eval()

# step 4 - load test data
transform  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_data   = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# step 5 - evaluate
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
print(f"correct: {correct}/{total}")