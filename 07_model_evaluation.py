# =========================================
# 07_model_evaluation.py
# Evaluasi Model Klasifikasi Penyakit Daun Mangga
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)

# =====================================================
# 8.1 Plot Kurva Akurasi & Loss
# =====================================================

tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]

index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

epochs = range(1, len(tr_acc) + 1)

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.scatter(index_loss + 1, val_lowest, s=150, label=f'Best Epoch {index_loss + 1}')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, label=f'Best Epoch {index_acc + 1}')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# =====================================================
# 8.2 Evaluasi Akurasi Model
# =====================================================

ts_length = len(test_df)
test_batch_size = max([
    ts_length // n
    for n in range(1, ts_length + 1)
    if ts_length % n == 0 and ts_length / n <= 80
])
test_steps = ts_length // test_batch_size

train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
test_score  = model.evaluate(test_gen,  steps=test_steps, verbose=1)

print("\n===== MODEL PERFORMANCE =====")
print(f"Train Accuracy      : {train_score[1]*100:.2f}%")
print(f"Validation Accuracy : {valid_score[1]*100:.2f}%")
print(f"Test Accuracy       : {test_score[1]*100:.2f}%")


# =====================================================
# 8.3 Prediksi Data Uji
# =====================================================

preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)


# =====================================================
# 8.4 Confusion Matrix & F1 Score
# =====================================================

classes = list(test_gen.class_indices.keys())

cm = confusion_matrix(test_gen.classes, y_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j, i, cm[i, j],
        horizontalalignment='center',
        color='white' if cm[i, j] > thresh else 'black'
    )

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


# =====================================================
# F1 Score Heatmap
# =====================================================

f1_scores = f1_score(test_gen.classes, y_pred, average=None)
f1_df = pd.DataFrame(f1_scores, index=classes, columns=['F1 Score'])

plt.figure(figsize=(8, 8))
sns.heatmap(f1_df, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('F1 Score per Class')
plt.ylabel('Class')
plt.xlabel('Score')
plt.show()


# =====================================================
# 8.5 Classification Report
# =====================================================

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    test_gen.classes,
    y_pred,
    target_names=classes
))
