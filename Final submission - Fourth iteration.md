# Fourth Iteration：
## Modifications：
- I utilized ResNet18 as the base model and added a fully connected layer at the end to adapt it to the four-class classification task. The pretrained=True parameter was used to leverage the pre-trained ResNet18 network.
- In this iteration, I implemented more extensive data augmentation techniques, including random horizontal flipping, random rotation, color enhancement, and random cropping.
- I also employed the Adam optimizer and incorporated the ReduceLROnPlateau learning rate scheduler, which adjusts the learning rate based on changes in validation accuracy. The learning rate is reduced only when the validation accuracy stops improving.
- After training, the best-performing model (the one with the highest accuracy) was saved and loaded for real-time inference. The model loading process is handled in the load_model() function.
- In the first segment of code, I used Streamlit to create an interface that displays real-time inference results. Users can activate the camera through the Streamlit interface to view the processed images. Additionally, I made further optimizations to the interface.
## Training code display：
```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define the improved CNN model (based on ResNet18)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # Use pre-trained ResNet18
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)  # Modify the last layer

    def forward(self, x):
        return self.base_model(x)

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),      # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color enhancement
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_path = 'c:/Users/Lenovo/Desktop/AI-final/hands/dataset/train'
val_path = 'c:/Users/Lenovo/Desktop/AI-final/hands/dataset/val'

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleCNN(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    best_accuracy = 0.0
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

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

        # Update learning rate
        scheduler.step(accuracy)

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
```

## Applying code display：
```
import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Define the improved CNN model (consistent with the training script)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=False)  # Do not use pre-trained weights
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Data preprocessing (consistent with the training script)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=4)
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))  # Load the best model
    model.eval()
    return model

model = load_model()

# Streamlit interface
st.title("Real-time Behavior Detection")
st.write("Use the camera to detect behaviors in real-time: Normal, Biting Nails, Picking Face, Biting Oral Mucosa")

# Start camera
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])  # Used to display video frames

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Explicitly specify the camera backend

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera")
        break

    # Convert frame to PIL image and preprocess
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0)

    # Use the model for prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output).item()

    # Convert prediction to behavior label
    labels = ['Normal', 'Biting Nails', 'Picking Face', 'Biting Oral Mucosa']
    label = labels[predicted]

    # Display the result on the frame
    frame = cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    FRAME_WINDOW.image(frame[:, :, ::-1])  # BGR to RGB

cap.release()
```
## Training results：
![6bbcf9d5a471c84e475a9e5b8ab810c](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/e581bc89-4062-421c-937e-be5c6a2d03bb)
![a22b294e9adfa094079e8c498e3a4c0](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/e9d00619-5267-4e36-9b29-038d36e2c566)
![4a006e9630056062e8a2a0a4f3c58bf](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/ffd12be1-f337-4985-955b-72f72ec6382f)
![d826925c6a5d7699329c9bed691b3d9](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/c2bc7d2b-27ad-42f0-9479-eb89bbf4ed26)
![639f0ac5e9ef4e266bbb4efc7edb42b](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/661771a8-2866-497f-9b57-ca54294aba42)

## Issues：
- Low Training Accuracy: The highest training accuracy achieved was only 58%. While this is an improvement over previous iterations, it remains insufficient. Possible reasons include dataset imbalance, overfitting or underfitting of the model, inadequate data augmentation, or suboptimal learning rates.
- Slow Camera Loading: The high resolution of the camera may cause processing delays.
- Streamlit Latency: Real-time display through Streamlit may introduce additional latency.
- Inaccurate Label Recognition: This issue may be related to model performance, as low accuracy naturally leads to unstable predictions. Additionally, the label display issue might stem from a mismatch between the class order in the code and the order used during training. It is necessary to check the class mapping in ImageFolder.

## Solutions：
- Enhance Data Augmentation and Preprocessing: Improve the model's generalization capability by refining data augmentation and preprocessing techniques.
- Ensure Consistent Label Order: Verify that the class order in the code matches the order used during training.
- Adopt a Lighter Model Architecture: Use a more lightweight model architecture, such as MobileNetV3, to accelerate both training and inference speeds.
- Freeze Layers of the Pre-trained Model: Reduce computational load by freezing certain layers of the pre-trained model.
- Optimize Data Loading Parameters: Adjust parameters such as batch size and the number of threads to optimize CPU performance.
- Introduce Mixed Precision Training: Implement mixed precision training to speed up training on the CPU.
- Optimize Real-time Detection: Improve frame rates through techniques such as frame skipping and asynchronous prediction.



