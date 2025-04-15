import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST handwritten digits dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data (convert to 0–1 range)
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels (0 to 9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),      # Convert 2D image to 1D vector
    Dense(128, activation='relu'),      # Hidden layer
    Dense(10, activation='softmax')     # Output layer (10 digits)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Show a sample prediction
index = 0
plt.imshow(x_test[index], cmap='gray')
plt.title("Actual Label: " + str(tf.argmax(y_test[index]).numpy()))
plt.show()

predicted = model.predict(x_test)
print("Predicted Label:", tf.argmax(predicted[index]).numpy())
