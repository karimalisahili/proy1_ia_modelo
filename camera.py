import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from time import sleep

CLASS_LABELS = {0: "rock", 1: "paper", 2: "scissors"}
# Load the trained model
model = load_model("rock_paper_scissors_mlp.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_label_name(class_index):
    return CLASS_LABELS.get(class_index, "Unknown")


def image_to_landmarks_list(image: np.ndarray):
    # Process the image with MediaPipe Hands
    results = hands.process(image)
    if results.multi_hand_landmarks:
        print("Hand landmarks detected.")
        # Extract the first hand's landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        # Convert landmarks to a list of (x, y, z) tuples
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        # Flatten the list of tuples
        landmarks = [coord for landmark in landmarks for coord in landmark]
        return landmarks
    else:
        return None


# Function to preprocess the frame for the model
def preprocess_frame(frame):
    landmarks = image_to_landmarks_list(frame)
    if landmarks is None:
        return None
    print(landmarks)
    landmarks = np.array(landmarks, dtype=np.float32)
    landmarks = np.expand_dims(landmarks, axis=0)
    return landmarks


# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
sleep(2)  # Allow time for the camera to warm up

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Define the region of interest (ROI) for the hand
    height, width, _ = frame.shape
    roi_x_start = width // 4
    roi_y_start = height // 4
    roi_x_end = roi_x_start + width // 2
    roi_y_end = roi_y_start + height // 2

    # Draw a rectangle on the frame to indicate the ROI
    cv2.rectangle(
        frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2
    )

    # Crop the ROI from the frame
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # Preprocess the cropped ROI
    input_frame = preprocess_frame(roi)

    # Make a prediction
    if input_frame is not None:
        print(input_frame)
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Convert the predicted class to its label
        label_name = get_label_name(predicted_class)

        # Display the prediction on the frame
        cv2.putText(
            frame,
            f"Prediction: {label_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # Show the frame with the rectangle
    cv2.imshow("Rock Paper Scissors - Real-Time", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
