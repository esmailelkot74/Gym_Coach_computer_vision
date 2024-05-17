import os

import cv2
import numpy as np
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    X = []
    y = []

    classes = ['Correct', 'Incorrect']
    for class_name in classes:
        class_dir = os.path.join("dataset", class_name)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (100, 100))  # Resize image
            X.append(image)
            y.append(class_name)

    y = np.array(y)
    y[y == 'Correct'] = 0
    y[y == 'Incorrect'] = 1
    y = to_categorical(y)

    return np.array(X), y

def train_classifier():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_datagen.fit(X_train)

    # Train the model
    history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                        steps_per_epoch=len(X_train) / 32, epochs=10, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy: ", accuracy)

    model.save("exercise_model")

if __name__ == "__main__":
    train_classifier()
