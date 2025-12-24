# =====================================
# Edge AI & AI-IoT Assignment
# =====================================

# Part 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.lite as tflite

# =====================================
# Task 1: Edge AI Prototype
# =====================================
print("===== TASK 1: Edge AI Image Classification Prototype =====")

# Load CIFAR-10 dataset (simulation for lightweight images)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Select a subset of classes for a lightweight Edge AI demo
selected_classes = [0, 1, 2]  # airplane, automobile, bird
train_mask = np.isin(y_train, selected_classes).flatten()
test_mask = np.isin(y_test, selected_classes).flatten()

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build lightweight CNN for Edge AI
edge_model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
edge_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Train model (simulation: few epochs for demo)
history = edge_model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

# Evaluate model
loss, accuracy = edge_model.evaluate(X_test, y_test)
print(f"Edge AI Model Test Accuracy: {accuracy:.2f}")

# Visualize sample predictions
sample_idx = np.random.choice(len(X_test), 5, replace=False)
for i in sample_idx:
    img = X_test[i]
    pred_class = np.argmax(edge_model.predict(img.reshape(1,32,32,3)))
    plt.imshow(img)
    plt.title(f"Predicted Class: {pred_class}")
    plt.show()

# =====================================
# Convert to TensorFlow Lite for Edge Deployment
# =====================================
converter = tflite.TFLiteConverter.from_keras_model(edge_model)
tflite_model = converter.convert()

# Save TFLite model
with open("edge_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TensorFlow Lite model saved as 'edge_model.tflite'")

# =====================================
# Task 2: AI-Driven IoT Concept (Smart Agriculture Simulation)
# =====================================
print("\n===== TASK 2: AI-IoT Concept - Smart Agriculture =====")

# Sensors for IoT setup
sensors = [
    "Soil Moisture Sensor",
    "Temperature Sensor",
    "Humidity Sensor",
    "Light Sensor",
    "pH Sensor"
]

# Proposed AI model: Random Forest / Lightweight Neural Network for yield prediction
ai_model_description = """
Input: Sensor readings (soil moisture, temp, humidity, light, pH)
Processing: AI model predicts crop yield in real-time
Output: Recommendations for irrigation, fertilization, and harvesting schedule
"""

# Display sensors and AI model description
print("Sensors needed for Smart Agriculture:", sensors)
print("AI Model Concept:\n", ai_model_description)

# Sample data flow diagram (textual representation)
data_flow_diagram = """
[Sensors] --> [Data Collection Node] --> [Preprocessing & Cleaning] --> [AI Model Prediction]
--> [Dashboard / Farmer Alerts] --> [Automated Irrigation & Fertilization Systems]
"""
print("Data Flow Diagram:\n", data_flow_diagram)
