## Prototype：
Initially, I decided to use a Convolutional Neural Network (CNN) as the foundational model for training. Therefore, I developed a prototype of a real-time object detection system based on the Faster R-CNN model. Below are the main functionalities and workflow of the code:
- Load a pre-trained Faster R-CNN model capable of detecting multiple objects in an image and outputting their bounding boxes and class labels.Set the model to evaluation mode (model.eval()) to avoid gradient computation during inference.
- Define image preprocessing operations to convert images into PyTorch's Tensor format. Use OpenCV to access the default camera.
- Draw detected object bounding boxes and labels on the image.
- Capture camera images frame by frame in real time.
- Convert the image from BGR format to RGB format and preprocess it.
- Feed the image into the model for inference, obtain detection results, and display them.

## Code display：
```
import torch 
import torchvision
import torchvision.transforms as T
import cv2

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Define image transformation operations
transform = T.Compose([
    T.ToTensor(),  # Convert image to Tensor
])

# Open the camera
cap = cv2.VideoCapture(0)  # 0 indicates the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to draw detection boxes
def draw_boxes(frame, predictions):
    for element in range(len(predictions[0]['boxes'])):
        box = predictions[0]['boxes'][element].cpu().numpy()  # Get bounding box
        label = predictions[0]['labels'][element].cpu().numpy()  # Get label
        score = predictions[0]['scores'][element].cpu().numpy()  # Get confidence score

        if score > 0.5:  # Only consider boxes with confidence greater than 0.5
            x_min, y_min, x_max, y_max = box
            label_name = f"Label: {label} | Score: {score:.2f}"

            # Draw bounding box and label on the image
            frame = cv2.rectangle(frame, 
                                  (int(x_min), int(y_min)), 
                                  (int(x_max), int(y_max)), 
                                  (0, 255, 0), 2)  # Green box
            frame = cv2.putText(frame, 
                                label_name, 
                                (int(x_min), int(y_min) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 255, 0), 
                                2, 
                                cv2.LINE_AA)
    return frame

while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to Tensor
    img_tensor = transform(img_rgb)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Perform inference without calculating gradients
    with torch.no_grad():
        predictions = model(img_tensor)

    # Draw detection boxes
    frame = draw_boxes(frame, predictions)

    # Display the processed frame
    cv2.imshow('Real-time Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
```
## Advantages：
- Real-time Performance: The code essentially meets the requirements for real-time camera capture and frame processing, fulfilling the need for real-time detection.
- Object Detection Capability: Faster R-CNN is a powerful object detection model capable of detecting multiple objects in an image and outputting their bounding boxes and class labels. It can be used to detect target regions such as hands and faces.
- Current Functionality: The existing code only displays detection results and does not provide a feedback mechanism.。

## Issues：
- Model Limitations: Faster R-CNN is a general-purpose object detection model, and its pre-trained version is based on the COCO dataset with 80 categories. This may limit its ability to directly detect specific behaviors such as "face-touching" or "nail-biting."
- Detection Speed: During testing, I observed that the detection speed was significantly slow. This is likely due to the high computational complexity of Faster R-CNN and the high input resolution.
