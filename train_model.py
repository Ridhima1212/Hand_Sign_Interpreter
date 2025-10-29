import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

DATA_DIR = "data"

# 1️⃣ Load all CSVs and merge
all_data = []
all_labels = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        label = file.split(".")[0]
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        all_data.append(df)
        all_labels += [label] * len(df)

X = pd.concat(all_data).values
y = np.array(all_labels)

print("Dataset loaded:")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])

# 2️⃣ Encode labels (A→0, B→1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 3️⃣ Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# 4️⃣ Build a simple neural network (MLP)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 5️⃣ Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32
)

# 6️⃣ Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# 7️⃣ Save model and label encoder
model.save("hand_sign_model.h5")
np.save("labels.npy", le.classes_)
print("✅ Model and labels saved successfully!")

# 8️⃣ Plot training accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
