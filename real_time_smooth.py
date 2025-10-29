import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# Load model and labels
model = tf.keras.models.load_model("hand_sign_model.h5")
labels = np.load("labels.npy")

# Init MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Queue to store recent predictions
window_size = 3
pred_window = deque(maxlen=window_size)
current_letter = ""

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
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            X = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(X, verbose=0)
            confidence = np.max(prediction)
            label = labels[np.argmax(prediction)]

            if confidence > 0.6:  # you can tune this
                pred_window.append(label)


            # Choose most frequent letter in last N frames
            if len(pred_window) > 0:
                most_common = Counter(pred_window).most_common(1)[0][0]
                current_letter = most_common
            else:
                current_letter = ""


    cv2.putText(image, f'Prediction: {current_letter}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Smoothed Sign Prediction", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
