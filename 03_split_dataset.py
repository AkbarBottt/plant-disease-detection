# ======================================
# 03_split_dataset.py
# Tahap 4 - Membagi dataset menjadi
# Train, Validation, dan Test
# ======================================

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =========================
# 1. Dataset Info
# =========================
data_dir = 'dataset/MangoLeafBD Dataset'
ds_name = 'Penyakit Daun Mangga'

print(f"Dataset Name : {ds_name}")
print(f"Dataset Path : {data_dir}\n")

# =========================
# 2. Membuat DataFrame
# =========================
def generate_data_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
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

print("Preview DataFrame:")
print(df.head(10), "\n")

# =========================
# 3. Fungsi Statistik
# =========================
def num_imgs(df, name='df'):
    print(f"Jumlah data {name} ada sebanyak {len(df)} gambar")


def num_of_classes(df, name='df'):
    print(f"jumlah jenis {name} memiliki {len(df['labels'].unique())} jenis penyakit")


def classes_count(df, name='df'):
    print(f"\njenis {name} pada dataset sebagai berikut:")
    print("=" * 70)
    for label in df['labels'].unique():
        count = len(df[df['labels'] == label])
        print(f"jenis '{label}' memiliki {count} gambar")
        print("-" * 70)


# =========================
# 4. Split Dataset
# =========================

# 70% Train, 30% sisa
train_df, dummy_df = train_test_split(
    df,
    train_size=0.7,
    shuffle=True,
    random_state=123
)

# 30% dibagi 50:50 → Validation & Test
valid_df, test_df = train_test_split(
    dummy_df,
    train_size=0.5,
    shuffle=True,
    random_state=123
)

# =========================
# 5. Hasil Split
# =========================
print("\n--- Jumlah Data ---")
num_imgs(train_df, 'Training ' + ds_name)
num_imgs(valid_df, 'Validation ' + ds_name)
num_imgs(test_df, 'Testing ' + ds_name)

print("\n--- Jumlah Kelas ---")
num_of_classes(train_df, 'Training ' + ds_name)
num_of_classes(valid_df, 'Validation ' + ds_name)
num_of_classes(test_df, 'Testing ' + ds_name)

print("\n--- Distribusi Kelas ---")
classes_count(train_df, 'Training ' + ds_name)
classes_count(valid_df, 'Validation ' + ds_name)
classes_count(test_df, 'Testing ' + ds_name)

print("\nSplit dataset selesai ✔")
