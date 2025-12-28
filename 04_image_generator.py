# ======================================
# 04_image_generator.py
# Tahap 5 - Image Data Generator
# ======================================

import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# 1. Load DataFrame dari tahap 4
# =========================

# NOTE:
# Cara aman: ulang generate dataframe + split
# (supaya file ini bisa berdiri sendiri)

from sklearn.model_selection import train_test_split

data_dir = 'dataset/MangoLeafBD Dataset'
ds_name = 'Penyakit Daun Mangga'

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

train_df, dummy_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=123)
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123)

# =========================
# 2. Parameter Gambar
# =========================
batch_size = 40
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# =========================
# 3. Test batch size khusus
# =========================
ts_length = len(test_df)

test_batch_size = max(
    sorted([
        ts_length // n
        for n in range(1, ts_length + 1)
        if ts_length % n == 0 and ts_length / n <= 80
    ])
)

test_steps = ts_length // test_batch_size

# =========================
# 4. Preprocessing Function
# =========================
def scalar(img):
    return img

# =========================
# 5. ImageDataGenerator
# =========================
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

ts_gen = ImageDataGenerator(
    preprocessing_function=scalar,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.4, 0.6],
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
)

# =========================
# 6. Generator Final
# =========================
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

valid_gen = ts_gen.flow_from_dataframe(
    valid_df,
    x_col='filepaths',
    y_col='labels',
    target_size=img_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

test_gen = ts_gen.flow_from_dataframe(
    test_df,
    x_col='filepaths',
    y_col='labels',
    target_size=img_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=test_batch_size
)

print("\nImage generator siap digunakan ✔")
print(f"Train batch size : {batch_size}")
print(f"Test batch size  : {test_batch_size}")
