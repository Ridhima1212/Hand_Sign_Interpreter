import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Setup folders
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
current_label = None
sample_count = 0

print("Press a key (A–Z) to start collecting data for that letter.")
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Save data if collecting
            if current_label is not None:
                csv_path = os.path.join(DATA_DIR, f"{current_label}.csv")
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)
                sample_count += 1
                cv2.putText(image, f"{current_label}: {sample_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Data Collection", image)
    key = cv2.waitKey(1) & 0xFF

    # Press ESC to exit
    if key == 27:
        break

    # Press any letter key to start/stop collecting
    elif 65 <= key <= 90 or 97 <= key <= 122:  # A–Z or a–z
        current_label = chr(key).upper()
        sample_count = 0
        print(f"Collecting data for: {current_label}")

cap.release()
cv2.destroyAllWindows()
hands.close()
