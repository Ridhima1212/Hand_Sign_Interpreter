import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import customtkinter as ctk
from PIL import Image, ImageTk

# ----- Model setup -----
model = tf.keras.models.load_model("hand_sign_model.h5")
labels = np.load("labels.npy")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----- Theme setup -----
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ----- Main App Window -----
app = ctk.CTk()
app.title("🤖 Real-Time AI Hand Sign Interpreter")
app.geometry("1000x700")
app.resizable(False, False)

# ----- Fonts -----
title_font = ("Segoe UI Semibold", 28)
label_font = ("Segoe UI", 16)
output_font = ("Consolas", 20)

# ----- Header -----
header_frame = ctk.CTkFrame(app, fg_color=("gray20"))
header_frame.pack(fill="x", pady=(10, 5))
ctk.CTkLabel(header_frame, text="🤖 AI Hand Sign Language Interpreter", font=title_font).pack(pady=10)

# ----- Video Display -----
video_frame = ctk.CTkFrame(app, corner_radius=15, border_color="deepskyblue", border_width=2)
video_frame.pack(pady=20)
video_label = ctk.CTkLabel(video_frame, text="")
video_label.pack(padx=10, pady=10)

# ----- Output Box -----
output_frame = ctk.CTkFrame(app, corner_radius=10, fg_color=("gray25"))
output_frame.pack(pady=10)
output_label = ctk.CTkLabel(output_frame, text="Output: ", font=output_font)
output_label.pack(padx=20, pady=10)

# ----- Global Variables -----
cap = None
hands = None
is_running = False
typed_text = ""
pred_window = deque(maxlen=5)
last_letter = ""
frame_counter = 0
stable_letter = ""

# ----- Frame Update Function -----
def update_frame():
    global typed_text, frame_counter, last_letter, stable_letter

    if not is_running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        app.after(10, update_frame)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            X = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(X, verbose=0)
            label = labels[np.argmax(prediction)]
            pred_window.append(label)

            if len(pred_window) > 0:
                stable_letter = Counter(pred_window).most_common(1)[0][0]

            if stable_letter == last_letter:
                frame_counter += 1
            else:
                frame_counter = 0
                last_letter = stable_letter

            if frame_counter == 8 and stable_letter != "":
                typed_text += stable_letter
                frame_counter = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((800, 500))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.configure(image=imgtk)
    video_label.image = imgtk

    output_label.configure(text=f"Output:  {typed_text}")
    app.after(30, update_frame)

# ----- Button Actions -----
def start_camera():
    global cap, hands, is_running
    if not is_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            output_label.configure(text="⚠️ Unable to access camera.")
            return
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        is_running = True
        output_label.configure(text="Camera started. Ready to detect ✋")
        update_frame()

def stop_camera():
    global cap, hands, is_running
    is_running = False
    if cap:
        cap.release()
        cap = None
    if hands:
        hands.close()
        hands = None
    video_label.configure(image="")
    output_label.configure(text="Output: (stopped)")

def clear_text():
    global typed_text
    typed_text = ""
    output_label.configure(text="Output: ")

def exit_app():
    stop_camera()
    app.destroy()

# ----- Buttons Section -----
button_frame = ctk.CTkFrame(app, fg_color=("gray22"))
button_frame.pack(pady=20)

button_style = {"width": 120, "height": 40, "corner_radius": 12, "font": label_font}
ctk.CTkButton(button_frame, text="▶ Start", command=start_camera, **button_style).grid(row=0, column=0, padx=10, pady=5)
ctk.CTkButton(button_frame, text="⏹ Stop", command=stop_camera, **button_style).grid(row=0, column=1, padx=10, pady=5)
ctk.CTkButton(button_frame, text="🧹 Clear", command=clear_text, **button_style).grid(row=0, column=2, padx=10, pady=5)
ctk.CTkButton(button_frame, text="❌ Exit", command=exit_app, fg_color="firebrick", hover_color="red4", **button_style).grid(row=0, column=3, padx=10, pady=5)

# ----- Footer -----
footer = ctk.CTkLabel(app, text="💡 Tip: Hold your sign steady for a second to add a letter", font=("Segoe UI", 13))
footer.pack(pady=(10, 5))

app.mainloop()
