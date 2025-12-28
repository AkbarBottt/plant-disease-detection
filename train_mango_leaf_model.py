# =========================================================
# TRAIN MANGO LEAF DISEASE MODEL
# Tahap 2 – 9 (Data → Train → Evaluate → Save Model)
# VERSI AMAN: Epoch 15 + Checkpoint + Save
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# TAHAP 2 – LOAD DATA & BUAT DATAFRAME
# =========================================================

DATASET_NAME = "Penyakit Daun Mangga"
DATASET_PATH = "dataset/MangoLeafBD Dataset"

def generate_data_paths(data_dir):
    filepaths = []
    labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                filepaths.append(os.path.join(folder_path, file))
                labels.append(folder)

    return filepaths, labels

filepaths, labels = generate_data_paths(DATASET_PATH)

df = pd.DataFrame({
    "filepaths": filepaths,
    "labels": labels
})

print("\nPreview DataFrame:")
print(df.head())
print(f"\nTotal gambar: {len(df)}")
print(f"Jumlah kelas: {df['labels'].nunique()}")

# =========================================================
# TAHAP 3 – EDA RINGKAS
# =========================================================

plt.figure(figsize=(10,5))
sns.countplot(data=df, x="labels")
plt.xticks(rotation=45)
plt.title("Distribusi Kelas Dataset")
plt.show()

# =========================================================
# TAHAP 4 – SPLIT DATAFRAME
# =========================================================

train_df, temp_df = train_test_split(
    df,
    train_size=0.7,
    shuffle=True,
    random_state=123,
    stratify=df["labels"]
)

valid_df, test_df = train_test_split(
    temp_df,
    train_size=0.5,
    shuffle=True,
    random_state=123,
    stratify=temp_df["labels"]
)

print("\nJumlah data:")
print("Train :", len(train_df))
print("Valid :", len(valid_df))
print("Test  :", len(test_df))

# =========================================================
# TAHAP 5 – IMAGE DATA GENERATOR
# =========================================================

img_size = (224, 224)
batch_size = 32

def scalar(img):
    return img

train_gen = ImageDataGenerator(
    preprocessing_function=scalar,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.4, 0.6],
    horizontal_flip=True,
    vertical_flip=True
)

test_gen = ImageDataGenerator(preprocessing_function=scalar)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col="filepaths",
    y_col="labels",
    target_size=img_size,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True
)

valid_data = test_gen.flow_from_dataframe(
    valid_df,
    x_col="filepaths",
    y_col="labels",
    target_size=img_size,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True
)

test_data = test_gen.flow_from_dataframe(
    test_df,
    x_col="filepaths",
    y_col="labels",
    target_size=img_size,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False
)

# =========================================================
# TAHAP 6 – VISUALISASI DATA TRAINING
# =========================================================

classes = list(train_data.class_indices.keys())
images, labels = next(train_data)

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(images[i] / 255)
    plt.title(classes[np.argmax(labels[i])])
    plt.axis("off")
plt.show()

# =========================================================
# TAHAP 7 – MODEL & TRAINING (AMAN)
# =========================================================

base_model = tf.keras.applications.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    pooling="max"
)

base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(),
    Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.016)
    ),
    Dropout(0.45),
    Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 🔐 CALLBACK AMAN
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "my_model_full-v1.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=50,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# =========================================================
# TAHAP 8 – EVALUASI MODEL
# =========================================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Valid")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Valid")
plt.title("Loss")
plt.legend()

plt.show()

test_loss, test_acc = model.evaluate(test_data)
print("\nTest Accuracy:", test_acc)

preds = model.predict(test_data)
y_pred = np.argmax(preds, axis=1)

cm = confusion_matrix(test_data.classes, y_pred)

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nClassification Report:\n")
print(classification_report(test_data.classes, y_pred, target_names=classes))

# =========================================================
# TAHAP 9 – SIMPAN MODEL FINAL
# =========================================================

model.save("my_model_full-v1-final.h5")
model.save_weights("my_model_weights-v1-final.h5")

print("\n✅ MODEL BERHASIL DISIMPAN DENGAN AMAN")
