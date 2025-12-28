# ======================================
# 05_visualize_training_data.py
# Tahap 6 - Visualisasi Dataset Training
# ======================================

import numpy as np
import matplotlib.pyplot as plt

# generator sudah dibuat di tahap 5
# supaya file ini berdiri sendiri, kita import ulang

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# =========================
# 1. Load Dataset & Generator
# =========================
data_dir = 'dataset/MangoLeafBD Dataset'

def generate_data_paths(data_dir):
    filepaths = []
    labels = []

    for fold in os.listdir(data_dir):
        foldpath = os.path.join(data_dir, fold)
        if os.path.isdir(foldpath):
            for file in os.listdir(foldpath):
                filepaths.append(os.path.join(foldpath, file))
                labels.append(fold)

    return filepaths, labels


def create_df(filepaths, labels):
    return pd.concat([
        pd.Series(filepaths, name='filepaths'),
        pd.Series(labels, name='labels')
    ], axis=1)


filepaths, labels = generate_data_paths(data_dir)
df = create_df(filepaths, labels)

train_df, _ = train_test_split(df, train_size=0.7, shuffle=True, random_state=123)

img_size = (224, 224)
batch_size = 40

def scalar(img):
    return img

tr_gen = ImageDataGenerator(
    preprocessing_function=scalar,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.4, 0.6],
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
)

train_gen = tr_gen.flow_from_dataframe(
    train_df,
    x_col='filepaths',
    y_col='labels',
    target_size=img_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

# =========================
# 2. Visualisasi Gambar
# =========================
g_dict = train_gen.class_indices     # {'Anthracnose': 0, ...}
classes = list(g_dict.keys())

images, labels = next(train_gen)

plt.figure(figsize=(20, 20))

for i in range(16):
    plt.subplot(4, 4, i + 1)

    image = images[i] / 255.0        # normalisasi untuk display
    plt.imshow(image)

    index = np.argmax(labels[i])
    class_name = classes[index]

    plt.title(class_name, color='blue', fontsize=15)
    plt.axis('off')

plt.tight_layout()
plt.show()
