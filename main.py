import os
# Suppress all logs and force CPU mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_WIDTH, IMG_HEIGHT = 30, 30
NUM_CATEGORIES = 43

def load_data(data_dir):
    images, labels = [], []
    for i in range(NUM_CATEGORIES):
        path = os.path.join(data_dir, str(i))
        for filename in os.listdir(path):
            if filename.startswith('.'): continue
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img.astype('float32') / 255.0)
                labels.append(i)
    return np.array(images), np.array(labels)

def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    if len(sys.argv) < 2: sys.exit("Usage: python traffic.py gtsrb")
    
    images, labels = load_data(sys.argv[1])
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.4)
    
    model = get_model()
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    main()
