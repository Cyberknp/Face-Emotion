import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm

# ---- Data paths ----
TRAIN_DIR = "images/train"
TEST_DIR = "images/test"


# ---- Helper Functions ----
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


# ---- Create DataFrames ----
train = pd.DataFrame()
train["image"], train["label"] = createdataframe(TRAIN_DIR)
print(train)

test = pd.DataFrame()
test["image"], test["label"] = createdataframe(TEST_DIR)
print(test)
print(test["image"])

# ---- Feature Extraction ----
train_features = extract_features(train["image"])
test_features = extract_features(test["image"])

# ---- Normalization ----
x_train = train_features / 255.0
x_test = test_features / 255.0

# ---- Encoding Labels ----
le = LabelEncoder()
le.fit(train["label"])
y_train = le.transform(train["label"])
y_test = le.transform(test["label"])

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# ---- Model Architecture ----
model = Sequential()
# Convolutional layers
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
# Fully connected layers
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
# Output layer
model.add(Dense(7, activation="softmax"))

# ---- Compilation ----
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---- Model Training ----
model.fit(
    x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test)
)

# ---- Save Model ----
model_json = model.to_json()
with open("emotiondetector.json", "w") as json_file:
    json_file.write(model_json)
model.save("emotiondetector.h5")

# ---- Load Model Example ----
# (Uncomment below if you want to load and use the model)
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("facialemotionmodel.h5")

# ---- Labels ----
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ---- Prediction Helper ----
def ef(image):
    img = load_img(image, grayscale=True)
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# ---- Example Prediction ----
image = "images/train/sad/42.jpg"
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = labels[pred.argmax()]
print("model prediction is", pred_label)
plt.imshow(img.reshape(48, 48), cmap="gray")
plt.show()

# Another prediction example
image = "images/train/fear/2.jpg"
print("original image is of fear")
img = ef(image)
pred = model.predict(img)
pred_label = labels[pred.argmax()]
print("model prediction is", pred_label)
plt.imshow(img.reshape(48, 48), cmap="gray")
plt.show()

# And so on for other image classes...
