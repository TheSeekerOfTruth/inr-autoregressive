import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# 3. Load MNIST Dataset (No Padding)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Define LeNet-5 Architecture for 28x28 Input
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Feature Extractor
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), # Output: 24x24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                             # Output: 12x12
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # Output: 8x8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                              # Output: 4x4
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=120),               # Changed from 16*5*5 to 16*4*4
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) 
        logits = self.classifier(x)
        return logits

model = LeNet5().to(device)

# 5. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Training Loop
print("Starting Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

# 7. Evaluation Loop
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\nTraining Complete! Final Test Accuracy: {accuracy:.2f}%")
# 8. Save the Checkpoint
checkpoint = {
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': accuracy,
}

checkpoint_path = "lenet5_mnist.pth"
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint successfully saved to {checkpoint_path}!")