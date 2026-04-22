import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Path to the trained CNN model
MODEL_PATH = "model.h5"

# List of emotion classes (change according to your model training)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load the trained CNN model
model = load_model(MODEL_PATH)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

print("Starting camera. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Preprocess the frame (modify according to your model input size and requirements)
    # Example: Resize to 48x48, convert to grayscale, expand dimensions, normalize
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = np.expand_dims(face, axis=-1)  # Add channel dimension if needed

    # Make prediction
    predictions = model.predict(face)
    max_index = np.argmax(predictions[0])
    emotion = emotion_labels[max_index]

    # Print predicted emotion in terminal
    print("Predicted Emotion:", emotion)

    # (Optional) Show the frame in a window
    cv2.imshow("Webcam Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
