import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# 1. Load Audio Features
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# 2. Prepare Dataset
data = []
labels = []

for file in os.listdir("audio"):
    if file.endswith(".wav"):
        file_path = os.path.join("audio", file)
        emotion = file.split("_")[0]  # e.g., happy_01.wav → happy
        features = extract_features(file_path)
        data.append(features)
        labels.append(emotion)

X = np.array(data)
y = LabelEncoder().fit_transform(labels)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Model
model = Sequential()
model.add(Dense(256, input_shape=(40,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(set(y)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# 6. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")
