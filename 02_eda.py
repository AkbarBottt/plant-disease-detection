# ======================================================
# 3 || EDA (Exploratory Data Analysis)
# ======================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

from plotly.subplots import make_subplots
import plotly.graph_objects as go



# ======================================================
# 3.1 || Tentukan jalur data dan nama dataset
# ======================================================

# === ASLI KAGGLE ===
# data_dir = '/kaggle/input/mango-leaf-disease-dataset/MangoLeafBD Dataset'

# === VERSI LOKAL (WINDOWS) ===
data_dir = r'dataset\MangoLeafBD Dataset'

ds_name = 'Penyakit Daun Mangga'

print(f"Dataset Name : {ds_name}")
print(f"Dataset Path : {data_dir}")


# ======================================================
# 3.2 || Membuat DataFrame untuk dataset
# ======================================================

def generate_data_paths(data_dir):
    
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)

        if not os.path.isdir(foldpath):
            continue

        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
            
    return filepaths, labels


filepaths, labels = generate_data_paths(data_dir)


def create_df(filepaths, labels):
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df


df = create_df(filepaths, labels)

print("\nPreview DataFrame:")
print(df.head(10))


# ======================================================
# 3.3 || Menampilkan jumlah contoh dalam dataset
# ======================================================

def num_of_examples(df, name='df'):
    print(f"jumlah gambar {name} memiliki sebanyak {df.shape[0]} gambar.")

num_of_examples(df, ds_name)


# ======================================================
# 3.4 || Menampilkan jumlah kelas dalam dataset
# ======================================================

def num_of_classes(df, name='df'):
    print(f"jumlah jenis {name} memiliki {len(df['labels'].unique())} jenis penyakit")

num_of_classes(df, ds_name)


# ======================================================
# 3.5 || Jumlah gambar di setiap kelas
# ======================================================

def classes_count(df, name='df'):
    print(f"jenis {name} pada dataset sebagai berikut:")
    print("=" * 70)
    for label in df['labels'].unique():
        num_class = len(df[df['labels'] == label])
        print(f"jenis '{label}' memiliki {num_class} gambar")
        print("-" * 70)

classes_count(df, ds_name)

print("\nCatatan: kelas 'Healthy' adalah daun mangga sehat.\n")


# ======================================================
# 3.6 || Visualisasi distribusi kelas
# ======================================================

colors = [
    '#494BD3', '#E28AE2', '#F1F481', '#79DB80',
    '#DF5F5F', '#69DADE', '#C2E37D', '#E26580'
]

def cat_summary_with_graph(dataframe, col_name):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Countplot', 'Persentase'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )

    fig.add_trace(
        go.Bar(
            x=dataframe[col_name].value_counts().index.astype(str),
            y=dataframe[col_name].value_counts().values,
            text=dataframe[col_name].value_counts().values,
            textposition='auto',
            marker=dict(color=colors),
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=dataframe[col_name].value_counts().index,
            values=dataframe[col_name].value_counts().values,
            marker=dict(colors=colors),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title={'text': col_name, 'x': 0.5},
        template='plotly_white'
    )

    fig.show()


cat_summary_with_graph(df, 'labels')


# ======================================================
# 3.7 || Memeriksa nilai Null
# ======================================================

def check_null_values(df, name='df'):
    num_null_vals = df.isnull().sum().sum()

    if num_null_vals == 0:
        print(f"dataset {name} tidak memiliki nilai null")
    else:
        print(f"dataset {name} memiliki {num_null_vals} nilai null")
        print(df.isnull().sum())

check_null_values(df, ds_name)


# ======================================================
# 3.8 || Visualisasi nilai Null
# ======================================================

msno.matrix(df)
plt.title('Distribusi Nilai yang Hilang (Missing Values)', fontsize=16)
plt.show()
