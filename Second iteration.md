# Second Iteration：
## Modifications：
- I decided to employ a custom model and training data to classify behaviors such as "nail-biting," "face-picking," and "oral mucosa biting."
- Furthermore, the system can be extended to detect other undesirable behaviors by augmenting the training data or modifying the model architecture. However, there are no readily available, publicly accessible, or standardized datasets specifically tailored for behaviors like "nail-biting," "face-picking," or "oral mucosa biting." These behaviors fall under specific, fine-grained action categories, necessitating the collection and annotation of data from scratch. To address this, I developed a program capable of automatically capturing images via a camera at predefined intervals and saving them to a designated folder, thereby facilitating the data collection process.

## Automatic Image Capture Code Display:
```
# Automatically capture images through the camera
import cv2
import os
import time

# Create folder
output_dir = 'C:/Users/Lenovo/Desktop/AI-final/hands/dataset/val/biting_nails'  # Modify the category as needed
os.makedirs(output_dir, exist_ok=True)

# Open the camera
cap = cv2.VideoCapture(0)

# Set image counter
count = 0

# Set the time interval for automatic saving (in seconds)
interval = 2  # Save an image every 2 seconds
last_save_time = time.time()

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture image")
        break

    # Display the image
    cv2.imshow('Press "q" to quit', frame)

    # Check if the save time interval has passed
    current_time = time.time()
    if current_time - last_save_time >= interval:
        # Save the image
        img_name = os.path.join(output_dir, f'image_{count}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
        last_save_time = current_time  # Update the last save time

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
```
## Image Display:
![d28b0bb8021d63bbc37aec855fcd525](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/d2e7abb2-1c71-44e3-82e4-cea566e75c4a)
![bfef3208aa25c9a934610ffb72cbde8](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/07df650c-f772-4047-8926-6fd2e7720b26)

## Bad behaviour for detection code presentation:
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

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):  # Modify to 4 classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleCNN(num_classes=4)  # Modify to 4 classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
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

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Real-time detection
model = SimpleCNN(num_classes=4)  # Modify to 4 classes
model.load_state_dict(torch.load('model.pth'))
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

    # Make predictions using the model
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    # Convert prediction to behavior label
    labels = ['Normal', 'Biting Nails', 'Picking Face', 'Biting Oral Mucosa']  # Added categories
    label = labels[predicted.item()]

    # Display the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
```

## Analysis of results：
- Training Loss: The training loss decreased from 1.847 to 0.0003, indicating that the model achieved a strong fit to the training data. However, the rapid decline in loss may suggest potential overfitting.
- Validation Accuracy: The validation accuracy fluctuated between 45% and 51.875%, showing no significant improvement. This indicates poor generalization performance on unseen data, as the model struggled to adapt effectively to the validation set.
- Model Architecture Issue: From the model structure output, it is evident that the input feature size of the fc1 layer is 200,704 (64 * 56 * 56), likely due to an incorrect calculation of the input image dimensions. Given an input image size of 224x224, after two MaxPool2d operations, the feature map size should be 56x56 (224 / 2 / 2 = 56). Consequently, the input feature size for fc1 should indeed be 64 * 56 * 56 = 200,704.
- Such an excessively large fully connected layer may result in an overwhelming number of parameters, increasing the risk of overfitting.
![5a20faa2a36df154565c63f059076c1](https://git.arts.ac.uk/24001444/AI-4-Media-Project-Chenyang-Wang/assets/1324/cb54cfd8-9ef4-4918-82bc-669046ecaddc)


## Issues：
- Dataset issues: The dataset may be small or unbalanced, resulting in the model failing to learn effective features. And the data enhancement is insufficient and the model is overfitted on the training set.
- Model structure problem: The amount of parameters in the fully connected layer is too large, which makes it easy to overfitting. Meanwhile, the model complexity may not be enough to capture the complex features in the data.
- Training parameter problems: the learning rate (lr=0.001) may not be appropriate. At the same time, the number of training epochs is small (num_epochs=10) and the model may not be sufficiently trained.



