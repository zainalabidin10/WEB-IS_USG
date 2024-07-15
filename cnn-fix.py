
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import save_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Input
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display, HTML
import cv2  # pastikan opencv sudah terinstal
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module='keras')


# Definisi jalur direktori yang berisi gambar
sdir = r'C:\Users\ZAINAL\Downloads\PIC\data_labeling\data_labeling'

# Daftar kosong untuk menyimpan jalur file gambar dan labelnya
filepaths = []
labels = []

# Mendapatkan daftar nama sub-direktori (kelas) dalam direktori sdir
classlist = os.listdir(sdir)

# Iterasi melalui setiap sub-direktori (kelas)
for klass in classlist:
    classpath = os.path.join(sdir, klass)  # Jalur lengkap ke sub-direktori
    if os.path.isdir(classpath):  # Memeriksa apakah ini adalah direktori
        flist = os.listdir(classpath)  # Mendapatkan daftar file dalam sub-direktori
        for f in flist:
            fpath = os.path.join(classpath, f)  # Jalur lengkap ke file gambar
            filepaths.append(fpath)  # Menambahkan jalur file ke daftar filepaths
            labels.append(klass)  # Menambahkan label kelas ke daftar labels

# Membuat Pandas Series dari daftar filepaths dan labels
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')

# Menggabungkan kedua Series menjadi dataframe df
df = pd.concat([Fseries, Lseries], axis=1)

# Mencetak lima baris pertama dari dataframe df
print(df.head())

# Menghitung dan mencetak jumlah gambar dalam setiap kelas (label)
print(df['labels'].value_counts())

# Proporsi data pelatihan, pengujian, dan validasi
train_split = 0.8
test_split = 0.1

# Menghitung proporsi data validasi
dummy_split = test_split / (1 - train_split)

# Membagi data menjadi data pelatihan dan "dummy" data pengujian
train_df, dummy_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123)

# Membagi "dummy" data pengujian menjadi data pengujian dan data validasi
test_df, valid_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)

# Mencetak panjang data dari setiap bagian
print('train_df length:', len(train_df), 'test_df length:', len(test_df), 'valid_df length:', len(valid_df))

# Menggunakan ImageDataGenerator untuk augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Membuat generator data untuk data pelatihan, validasi, dan pengujian
train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                                    target_size=(128, 128), class_mode='categorical', batch_size=8)  # Mengurangi ukuran batch

valid_generator = valid_datagen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                                    target_size=(128, 128), class_mode='categorical', batch_size=8)  # Mengurangi ukuran batch

test_generator = test_datagen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                                  target_size=(128, 128), class_mode='categorical', batch_size=8)  # Mengurangi ukuran batch

# Membuat objek Input sebagai lapisan pertama dalam model
input_shape = (128, 128, 3)
inputs = Input(shape=input_shape)

# Membangun model Sequential
model_cnn = tf.keras.Sequential([
    inputs,
    Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2, strides=2),
    Dropout(rate=0.3),
    Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2, strides=2),
    Dropout(rate=0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(rate=0.3),
    Dense(2, activation='softmax')
])

# Mengkompilasi model dengan pengoptimal Adam, fungsi loss categorical_crossentropy, dan metrik akurasi
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Mencetak ringkasan model
model_cnn.summary()

# Mencetak nama model
model_name = 'Model_CNNs_domba'
print("Building model with", model_name)

# Callback untuk menyimpan model terbaik selama pelatihan
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# Melatih model
history = model_cnn.fit(train_generator, 
                        validation_data=valid_generator, 
                        epochs=10, 
                        callbacks=[checkpoint])

# Mencetak hasil pelatihan
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

# Menguji model terbaik
best_model = load_model('best_model.keras')
test_loss, test_acc = best_model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Membuat prediksi dengan model terbaik
y_pred = best_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Mencetak laporan klasifikasi
print('Classification Report')
print(classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

# Membuat confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Menyimpan model
model.save('model_cnn-FIX.h5')

