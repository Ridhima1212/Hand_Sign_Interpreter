import cv2
import mediapipe as mp 
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# Load model and labels
model = tf.keras.models.load_model("hand_sign_model.h5")
labels = np.load("labels.npy")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Variables for word building
typed_text = ""
last_letter = ""
frame_counter = 0
stable_letter = ""
window_size = 5
pred_window = deque(maxlen=window_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert to array and predict
            X = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(X, verbose=0)
            pred_label = labels[np.argmax(prediction)]
            pred_window.append(pred_label)

            # Smoothing: take most common letter in recent frames
            if len(pred_window) > 0:
                most_common = Counter(pred_window).most_common(1)[0][0]
                stable_letter = most_common
            else:
                stable_letter = ""

            # Add letter if it stays stable for a few frames
            if stable_letter == last_letter:
                frame_counter += 1
            else:
                frame_counter = 0
                last_letter = stable_letter

            # When a letter remains steady for N frames, add it to the text
            if frame_counter == 8 and stable_letter != "":
                typed_text += stable_letter
                frame_counter = 0

            # Display prediction
            cv2.putText(image, f'Prediction: {stable_letter}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Display the typed text
    cv2.putText(image, f'Typed: {typed_text}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Sign Language Word Builder (Stable)", image)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # Spacebar
        typed_text += " "
    elif key == 8:  # Backspace
        typed_text = typed_text[:-1]
    elif key == ord('c') or key == ord('C'):  # Clear
        typed_text = ""

cap.release()
cv2.destroyAllWindows()
hands.close()
