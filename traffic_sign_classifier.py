import os, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Unzip dataset
with zipfile.ZipFile("traffic_Data.zip", "r") as zip_ref:
    zip_ref.extractall(".")

dataset_path = "./DATA"
labels_csv_path = "./labels.csv"

# Load labels
labelfile = pd.read_csv(labels_csv_path)

# Sample Image Visualization
img_path = os.path.join(dataset_path, "11", "011_0011.png")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.savefig("visualizations/sample_image.png")

# Prepare Datasets
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Map class labels
class_names = []
for idx in train_ds.class_names:
    class_names.append(labelfile['Name'][int(idx)])

# Model Definition
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
])

model = Sequential([
    data_augmentation,
    Rescaling(1./255),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(len(labelfile), activation='softmax')
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

# Train model
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

# Save plots
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.savefig("visualizations/loss_plot.png")

plt.clf()

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.savefig("visualizations/accuracy_plot.png")

# Save model
model.save("models/model.h5")