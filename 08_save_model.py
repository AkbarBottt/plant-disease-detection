# =========================================
# 08_save_model.py
# Menyimpan Model Klasifikasi Daun Mangga
# =========================================

# Simpan bobot (weights) saja
model.save_weights('my_model_weights-v1.h5')

# Simpan SELURUH model (arsitektur + weights)
model.save('my_model_full-v1.h5')

print("Model berhasil disimpan:")
print("- my_model_weights-v1.h5  (bobot saja)")
print("- my_model_full-v1.h5     (model lengkap)")
