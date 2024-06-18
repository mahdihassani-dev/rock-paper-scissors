import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create directories for storing the images
os.makedirs('rock', exist_ok=True)
os.makedirs('paper', exist_ok=True)
os.makedirs('scissors', exist_ok=True)

# Function to preprocess landmarks and save as image
def preprocess_and_save_landmarks(landmarks, frame, save_path, img_size=128):
    # Get the frame dimensions
    h, w, _ = frame.shape

    # Get the bounding box of the hand
    x_min = min([landmark.x for landmark in landmarks])
    x_max = max([landmark.x for landmark in landmarks])
    y_min = min([landmark.y for landmark in landmarks])
    y_max = max([landmark.y for landmark in landmarks])

    # Calculate the center and size of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    bbox_size = max(x_max - x_min, y_max - y_min)

    # Calculate the scaling factor and translate the landmarks
    scale = img_size / bbox_size
    black_bg = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for landmark in landmarks:
        x = int((landmark.x - x_center) * scale + img_size / 2)
        y = int((landmark.y - y_center) * scale + img_size / 2)
        cv2.circle(black_bg, (x, y), 2, (255, 0, 0), -1)

    cv2.imwrite(save_path, black_bg)

# Open the webcam
cap = cv2.VideoCapture(0)

gesture = None
img_count = 0
save_dir = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # Display instructions
    if gesture is None:
        cv2.putText(frame, "Press 'r' for Rock, 'p' for Paper, 's' for Scissors", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Capturing {gesture} images: {img_count}/5000", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if results.multi_hand_landmarks and gesture is not None and img_count < 5000:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the landmarks
            landmarks = hand_landmarks.landmark

            # Save the preprocessed landmarks as an image
            save_path = os.path.join(save_dir, f"{gesture}_{img_count:04d}.png")
            preprocess_and_save_landmarks(landmarks, frame, save_path)
            img_count += 1

            if img_count >= 5000:
                gesture = None
                img_count = 0

    # Display the frame
    cv2.imshow('Capture Hand Gestures', frame)

    # Handle keypress events
    key = cv2.waitKey(1)
    if key == ord('r'):
        gesture = 'rock'
        save_dir = 'rock'
    elif key == ord('p'):
        gesture = 'paper'
        save_dir = 'paper'
    elif key == ord('s'):
        gesture = 'scissors'
        save_dir = 'scissors'
    elif key == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
