import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4,
)

# Load the pre-trained model
model = load_model("rock_paper_scissors_model.h5")

REV_CLASS_MAP = {
    0: "paper",
    1: "rock",
    2: "scissors",
}

# Load AI move images
ai_images = {
    "rock": cv2.imread("images/rock.png"),
    "paper": cv2.imread("images/paper.png"),
    "scissors": cv2.imread("images/scissors.png"),
}

def mapper(val):
    return REV_CLASS_MAP[val]

# Function to preprocess landmarks
def preprocess_landmarks(landmarks, img_size=128):
    x_min = min([landmark.x for landmark in landmarks])
    x_max = max([landmark.x for landmark in landmarks])
    y_min = min([landmark.y for landmark in landmarks])
    y_max = max([landmark.y for landmark in landmarks])
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    bbox_size = max(x_max - x_min, y_max - y_min)
    
    scale = img_size / bbox_size
    black_bg = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for landmark in landmarks:
        x = int((landmark.x - x_center) * scale + img_size / 2)
        y = int((landmark.y - y_center) * scale + img_size / 2)
        cv2.circle(black_bg, (x, y), 2, (255, 0, 0), -1)
    
    return black_bg

last_known_position = None
counter = 0
yHigh = 1000
yChanges = 0
y = 0
user_move_name = None
ai_move_name = None

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    half_width = width // 2
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x, y = int(wrist.x * width), int(wrist.y * height)
            last_known_position = (x, y)

        if y < yHigh:
            yHigh = y
        else:
            yChanges = y - yHigh
        if yChanges > 170:
            counter += 1
            yChanges = 0
            yHigh = 1000
            
        if counter == 3:
            preprocessed_landmarks = preprocess_landmarks(landmarks)
            preprocessed_landmarks = cv2.resize(preprocessed_landmarks, (128, 128))
            cv2.imwrite("landmarks.png", preprocessed_landmarks)
            preprocessed_landmarks = preprocessed_landmarks / 255.0
            preprocessed_landmarks = np.expand_dims(preprocessed_landmarks, axis=0)
            
            pred = model.predict(preprocessed_landmarks, verbose=0)[0]
            move_code = np.argmax(pred)
            user_move_name = mapper(move_code)
            
            # Save the frame
            cv2.imwrite("captured_frame.png", frame)
            
            if user_move_name == 'paper':
                ai_move_name = 'scissors'
            elif user_move_name == 'scissors':
                ai_move_name = 'rock'
            elif user_move_name == 'rock':
                ai_move_name = 'paper'
            print(f"User: {user_move_name}, AI: {ai_move_name}")
            counter = 0

    # Prepare the AI move display
    if ai_move_name:
        ai_move_display = ai_images[ai_move_name]
        ai_move_display = cv2.resize(ai_move_display, (half_width, height))
    else:
        ai_move_display = np.zeros((height, half_width, 3), np.uint8)

    combined_display = np.hstack((frame[:, :half_width], ai_move_display))
    

    # Display the counter in the center of the screen
    cv2.putText(
        combined_display,
        f"Counter: {counter}",
        (width // 2 - 100, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Display the human move
    if user_move_name:
        cv2.putText(
            combined_display,
            f"Your move: {user_move_name}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Display the AI move
    if ai_move_name:
        cv2.putText(
            combined_display,
            f"AI move: {ai_move_name}",
            (half_width + 20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Rock-Paper-Scissors Game", combined_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
