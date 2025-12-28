# Plant Disease Detection (Python)

Project ini mendeteksi penyakit tanaman mangga menggunakan **image classification** dengan CNN (TensorFlow/Keras).

## Isi Project

- `01_necessaries.py` – import library & fungsi dasar
- `02_eda.py` – eksplorasi dataset
- `03_split_dataset.py` – membagi dataset train/test
- `04_image_generator.py` – preprocessing & augmentasi gambar
- `05_visualize_training_data.py` – visualisasi dataset
- `06_model_training.py` – training model CNN
- `07_model_evaluation.py` – evaluasi model (accuracy, F1-score, confusion matrix)
- `08_save_model.py` – menyimpan model hasil training
- `10_predict_image.py` – prediksi gambar baru
- `train_mango_leaf_model.py` – script utama untuk training model

## Requirement

```bash
pip install -r requirements.txt
```

Cara Pakai

1. Pastikan dataset ada di folder input/
2. Jalankan training:

```bash
python train_mango_leaf_model.py
```

3. Lakukan evaluasi:

```bash
python 07_model_evaluation.py
```

4. Prediksi gambar baru:

```bash
python 10_predict_image.py
```

Catatan
Dataset dan model besar tidak diikutkan di repo.
Silakan download dataset dari sumber masing-masing dan letakkan di folder dataset/.
