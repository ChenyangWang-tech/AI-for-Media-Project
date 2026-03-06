# Third Iteration：
## Modifications：
- Model Architecture Adjustment:A global average pooling layer (`nn.AdaptiveAvgPool2d`) was incorporated to reduce the number of parameters in the fully connected layer, thereby mitigating the risk of overfitting.
- Data Augmentation:Random horizontal flipping and random rotation were introduced during data preprocessing to enhance the model's generalization capability.
- Learning Rate Scheduler: The `torch.optim.lr_scheduler.StepLR` was employed to dynamically adjust the learning rate, preventing it from becoming excessively high or low during the training process.
- Resolution of Warning Messages: To address potential security risks, the `weights_only=True` parameter was explicitly set when loading model weights.
- Increase in Training Epochs:The number of training epochs was increased from 10 to 20 to ensure more thorough training of the model.

## Code display ：
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import cv2
import os

# Check the current working directory
print("Current working directory:", os.getcwd())

# If the working directory is incorrect, switch to the correct directory
os.chdir("C:/Users/Lenovo/Desktop/AI-final")

# Define the improved CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):  # Modified for 4 classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)  # Global average pooling
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),     # Random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleCNN(num_classes=4)  # Modified for 4 classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Validation Accuracy: {100 * correct / total}%')
        scheduler.step()  # Update learning rate

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Real-time detection
model = SimpleCNN(num_classes=4)  # Modified for 4 classes
model.load_state_dict(torch.load('model.pth', weights_only=True))  # Resolve warning
model.eval()

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    input_tensor = transform(pil_img).unsqueeze(0)

    # Perform prediction using the model
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    # Map prediction to behavior label
    labels = ['Normal', 'Biting Nails', 'Picking Face', 'Biting Oral Mucosa']  # Added classes
    label = labels[predicted.item()]

    # Display the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
```
## Training results：
![5c098af2e6de436c978bdd470413690](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/9d5c6d8c-808e-4005-8c54-7e87d17031b3)
![5c098af2e6de436c978bdd470413690](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/626eb58c-034f-4c0d-8c52-2ad56b96b57a)

![0d5c4b97c17eb00372eb8a45c85e1c5](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/a8aaebdf-6363-4a3b-ba84-bab159303741)

## Issues：
- Training loss decreases too fast: The training loss decreases rapidly from 1.97 in the 1st epoch to 0.00067 in the 10th epoch, which indicates that the model may be overfitting the training data. The reason for overfitting may be that the model is too complex, or the amount of training data is insufficient.
- Validation accuracy is unstable: the validation accuracy reaches 67.5% in the 1st epoch, but then it drops to around 50%~53% with no significant improvement. This indicates that the model has poor generalisation ability on the validation set, possibly due to overfitting or data distribution problems.
- Low validation accuracy: the validation accuracy is around 50%~53%, which is close to random guessing (25% for 4 categories), indicating that the model fails to learn the features of the data effectively.

## Solutions：
- Ensure that the number of images in each category is approximately the same, or use class_weight to balance the loss function. 
- Insufficient data volume Solution: Increase the data volume, or use data enhancement (e.g., rotate, flip, crop, etc.) to expand the dataset.
- Model complexity problemSolution: The model I built is a simple CNN, which may not be able to capture complex features. So I try to use a more complex model (e.g. ResNet, EfficientNet, etc.) or use Transfer Learning.
- Solution to learning rate problem: Try to reduce the learning rate, or use a learning rate scheduler (e.g. ReduceLROnPlateau).


