# First Iteration：
## Modifications：
- Loading Gesture Recognition and Facial Detection: I decided to use MediaPipe, a pre-trained model capable of detecting hand and facial keypoints with low computational cost and minimal latency, offering higher accuracy.
- Behavioral Prompt: Added a detect_behavior function to detect user behaviors (e.g., "nail-biting" or "face-touching"). By analyzing the positional relationships between hand and facial keypoints, the system identifies specific behaviors.
- If a specific behavior is detected (e.g., hand near face), a prompt (e.g., "Probably chewing nails!") is displayed on the video frame.

## Code display：
```
import torch 
import torchvision
import torchvision.transforms as T
import cv2
import mediapipe as mp

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Define image transformation operations
transform = T.Compose([T.ToTensor()])  # Convert image to Tensor

# Initialize MediaPipe for gesture recognition
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

# Detect behavior (e.g., nail biting, face touching, etc.)
def detect_behavior(frame, hands_results, face_results):
    behavior = ""
    
    # Check if any hand is detected
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # If fingers are near the face area, alert for "nail biting" (this is a simple example, conditions need to be adjusted based on the actual scenario)
            for landmark in hand_landmarks.landmark:
                # Check the finger positions (e.g., detect the distance between fingers and face)
                if landmark.x > 0.5 and landmark.y > 0.5:
                    behavior = "Probably chewing nails!"
                    break

    # Check if any face is detected
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Check if a hand is touching the face
            for landmark in face_landmarks.landmark:
                if landmark.x > 0.5 and landmark.y > 0.5:
                    behavior = "Probably snapping faces!"
                    break

    return behavior

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

    # Use MediaPipe for gesture recognition and face detection
    results_hands = hands.process(img_rgb)
    results_face = face_mesh.process(img_rgb)

    # Detect behavior
    behavior = detect_behavior(frame, results_hands, results_face)
    if behavior:
        cv2.putText(frame, behavior, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the processed frame
    cv2.imshow('Real-time Object Detection and Behavior Alert', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
```

## Issues：
- Higher hardware requirements.
- Target detection using the pre-trained Faster R-CNN model is easy to misjudge by the relative position of the hand and face only.

