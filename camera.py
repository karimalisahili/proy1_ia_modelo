import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import threading
import random

CLASS_LABELS = {0: "Piedra", 1: "Papel", 2: "Tijera"}
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

is_playing = False


def get_label_name(class_index: int):
    """Convert class index to label name"""
    return CLASS_LABELS.get(class_index, "Unknown")


def image_to_landmarks_list(image: np.ndarray):
    """Convert an image to a list of hand landmarks"""
    results = hands.process(image)
    if results.multi_hand_landmarks:
        # Extract the first hand's landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        # Convert landmarks to a list of (x, y, z) tuples
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        # Flatten the list of tuples
        landmarks = [coord for landmark in landmarks for coord in landmark]
        return landmarks
    else:
        return None


def get_roi_shape(frame: np.ndarray):
    """Get the shape of the region of interest (ROI)"""
    height, width, _ = frame.shape
    roi_x_start = width // 4
    roi_y_start = height // 4
    roi_x_end = roi_x_start + width // 2
    roi_y_end = roi_y_start + height // 2
    return (roi_x_start, roi_y_start, roi_x_end, roi_y_end)


def draw_roi(frame: np.ndarray, roi_shape: tuple):
    """Draw a rectangle on the frame to indicate the ROI"""
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_shape
    cv2.rectangle(
        frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2
    )


def preprocess_frame(frame: np.ndarray):
    """Function to preprocess the frame for the model"""
    landmarks = image_to_landmarks_list(frame)
    if landmarks is None:
        return None
    landmarks = np.array(landmarks, dtype=np.float32)
    landmarks = np.expand_dims(landmarks, axis=0)
    return landmarks


def detect_hand(frame: np.ndarray, roi_shape: tuple):
    """Detect hand in the frame and return the Label"""
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_shape
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Convert to landmark list and pass to model
    input_frame = preprocess_frame(roi)
    if input_frame is not None:
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Convert the predicted class to its label
        label_name = get_label_name(predicted_class)

        # Display the prediction on the frame
        cv2.putText(
            frame,
            f"Prediccion: {label_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return label_name

    return None


def draw_controls(frame: np.ndarray):
    """Draw controls on the frame"""
    cv2.putText(
        frame,
        "Presiona Espacio para jugar",
        (10, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )

    cv2.putText(
        frame,
        "Presiona 'q' para salir",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )


def countdown_and_detect(frame: np.ndarray, roi_shape: tuple):
    """Function to handle the countdown in a separate thread"""
    global is_playing
    height, width, _ = frame.shape
    for i in range(3, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                return

            # Draw the ROI rectangle
            draw_roi(frame, roi_shape)

            # Display countdown number on the frame
            cv2.putText(
                frame,
                str(i),
                (width // 2 - 25, height // 2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                5,
            )
            cv2.imshow("Piedra Papel o Tijera", frame)
            cv2.waitKey(1)

    # After countdown, detect hand gesture
    prediction = detect_hand(frame, roi_shape)
    if prediction is None:
        start_time = time.time()
        while time.time() - start_time < 2:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                return

            # Draw the ROI rectangle
            draw_roi(frame, roi_shape)

            # No se encontró una mano
            cv2.putText(
                frame,
                "No se encontro una mano",
                (width // 2 - 200, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Piedra Papel o Tijera", frame)
            cv2.waitKey(1)
    else:
        # Computer generates a random choice
        computer_choice = random.randint(0, 2)
        computer_label = get_label_name(computer_choice)

        # Determine the winner
        if prediction == computer_label:
            result = "Empate"
        elif (
            (prediction == "Piedra" and computer_label == "Tijera")
            or (prediction == "Papel" and computer_label == "Piedra")
            or (prediction == "Tijera" and computer_label == "Papel")
        ):
            result = "Ganaste"
        else:
            result = "Perdiste"

        # Display the result
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                return

            # Draw the ROI rectangle
            draw_roi(frame, roi_shape)

            # Display the result message
            cv2.putText(
                frame,
                f"Tu: {prediction} | PC: {computer_label}",
                (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                result,
                (width // 2 - 120, height // 2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0) if result == "Ganaste" else (0, 0, 255),
                3,
            )
            cv2.imshow("Piedra Papel o Tijera", frame)
            cv2.waitKey(1)  # Allow OpenCV to process events

    is_playing = False


# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
time.sleep(1)  # Allow time for the camera to warm up

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
    roi_shape = get_roi_shape(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" ") and not is_playing:
        # Start the countdown in a separate thread
        is_playing = True
        threading.Thread(
            target=countdown_and_detect,
            args=(frame.copy(), roi_shape),
        ).start()

    if not is_playing:
        draw_controls(frame)
        draw_roi(frame, roi_shape)
        cv2.imshow("Piedra Papel o Tijera", frame)

    # Break the loop if 'q' is pressed
    if key in [ord("q"), ord("Q")]:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
