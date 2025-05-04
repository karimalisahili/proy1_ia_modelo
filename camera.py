import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

CLASS_LABELS = {0: "rock", 1: "paper", 2: "scissors"}
# Load the trained model
model = load_model('rock_paper_scissors_cnn.h5')

def get_label_name(class_index):
    return CLASS_LABELS.get(class_index, "Unknown")

# Function to preprocess the frame for the model
def preprocess_frame(frame, target_size=(150, 150)):
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize the frame to the target size
    img = Image.fromarray(frame_rgb).resize(target_size)
    # Convert the image to a NumPy array and normalize pixel values
    img_array = np.array(img) / 255.0
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

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

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Make a prediction
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Convert the predicted class to its label
    label_name = get_label_name(predicted_class)

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {label_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Rock Paper Scissors - Real-Time', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()