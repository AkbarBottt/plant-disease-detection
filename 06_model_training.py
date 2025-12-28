# =========================================================
# 06 || MODEL TRAINING - EfficientNetB7
# =========================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import os

# =========================================================
# PATH DATASET (SESUAIKAN)
# =========================================================

BASE_DIR = "dataset"   # contoh: dataset/train , dataset/valid
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")

# =========================================================
# IMAGE GENERATOR
# =========================================================

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

valid_gen = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =========================================================
# 7.1 || STRUKTUR MODEL
# =========================================================

class_count = train_gen.num_classes
img_shape = (img_size[0], img_size[1], 3)

base_model = tf.keras.applications.efficientnet.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=img_shape,
    pooling='max'
)

base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),

    Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006)
    ),

    Dropout(0.45),
    Dense(class_count, activation='softmax')
])

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

plot_model(
    model,
    to_file="model_structure.png",
    show_shapes=True
)

# =========================================================
# 7.2 || EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================================================
# 7.3 || TRAINING
# =========================================================

epochs = 50

history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=valid_gen,
    callbacks=[early_stopping]
)

# =========================================================
# SAVE MODEL
# =========================================================

model.save("efficientnetb7_mango_leaf.h5")
print("✅ Model berhasil disimpan")
