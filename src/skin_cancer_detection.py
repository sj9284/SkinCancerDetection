import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from PIL import Image
np.random.seed(42)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.cm

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
IMG_DIR = os.path.join(DATA_DIR, 'All Images')
MODEL_DIR = os.path.join(BASE_DIR, '../models')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 3: Load and preprocess dataset
SIZE = 32
skin_df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
le = LabelEncoder()
le.fit(skin_df['dx'])
print("Classes:", list(le.classes_))  # Should show ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
skin_df['label'] = le.transform(skin_df["dx"])
print("Sample of dataset:\n", skin_df.sample(10))

# Step 4: Visualize data distribution
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Diagnosis Distribution')

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count')
ax2.set_title('Sex Distribution')

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar', ax=ax3)
ax3.set_ylabel('Count')
ax3.set_title('Localization Distribution')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'data_distribution.png'))
plt.close()

print("Diagnosis distribution:\n", skin_df['dx'].value_counts())

# Step 5: Balance dataset
df_0 = skin_df[skin_df['label'] == 0]  # akiec
df_1 = skin_df[skin_df['label'] == 1]  # bcc
df_2 = skin_df[skin_df['label'] == 2]  # bkl
df_3 = skin_df[skin_df['label'] == 3]  # df
df_4 = skin_df[skin_df['label'] == 4]  # mel
df_5 = skin_df[skin_df['label'] == 5]  # nv
df_6 = skin_df[skin_df['label'] == 6]  # vasc

n_samples = 500  # Balance to 500 samples per class
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, df_2_balanced, df_3_balanced,
                              df_4_balanced, df_5_balanced, df_6_balanced])
print("Balanced diagnosis distribution:\n", skin_df_balanced['dx'].value_counts())

# Step 6: Load and preprocess images
image_path = {os.path.splitext(filename)[0]: os.path.join(IMG_DIR, filename) for filename in os.listdir(IMG_DIR)}
skin_df_balanced['path'] = skin_df_balanced['image_id'].map(image_path.get)
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))

X = np.asarray(skin_df_balanced['image'].tolist())
X = X / 255.0
Y = skin_df_balanced['label']
Y_cat = to_categorical(Y, num_classes=7)
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

# Step 7: Visualize sample images
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))
for n_axs, (type_name, type_rows) in zip(m_axs, skin_df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'], cmap=matplotlib.cm.Greys_r)
        c_ax.axis('off')
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_images.png'))
plt.close()

# Step 8: CNN Model
cnn = Sequential()
cnn.add(Conv2D(512, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
cnn.add(BatchNormalization())
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Dropout(0.3))
cnn.add(Conv2D(256, (3, 3), activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Dropout(0.3))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Dropout(0.3))
cnn.add(Flatten())
cnn.add(Dense(64))
cnn.add(Dense(7, activation='softmax', kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))

cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.summary()
batch_size = 16
epochs = 60
history_cnn = cnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
score_cnn = cnn.evaluate(x_test, y_test)
print('CNN Test accuracy:', score_cnn[1])

# Step 9: ANN Model
ann = Sequential()
ann.add(Flatten(input_shape=(SIZE, SIZE, 3)))
ann.add(Dense(512, activation="relu"))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))
ann.add(Dense(256, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))
ann.add(Dense(128, activation='relu'))
ann.add(BatchNormalization())
ann.add(Dropout(0.3))
ann.add(Dense(64))
ann.add(Dense(7, activation='softmax', kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))

ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ann.summary()
history_ann = ann.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
score_ann = ann.evaluate(x_test, y_test)
print('ANN Test accuracy:', score_ann[1])

# Step 10: FNN Model
fnn = Sequential()
fnn.add(Flatten(input_shape=(SIZE, SIZE, 3)))
fnn.add(Dense(512, activation="relu"))
fnn.add(BatchNormalization())
fnn.add(Dropout(0.3))
fnn.add(Dense(256, activation="relu"))
fnn.add(BatchNormalization())
fnn.add(Dropout(0.3))
fnn.add(Dense(128, activation="relu"))
fnn.add(BatchNormalization())
fnn.add(Dropout(0.3))
fnn.add(Dense(64, activation="relu"))
fnn.add(Dense(7, activation='softmax', kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))

fnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fnn.summary()
history_fnn = fnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
score_fnn = fnn.evaluate(x_test, y_test)
print('FNN Test accuracy:', score_fnn[1])

# Step 11: Compare accuracies
meth = ["CNN", "ANN", "FNN"]
acc_res = [score_cnn[1] * 100, score_ann[1] * 100, score_fnn[1] * 100]
plt.bar(meth, acc_res)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_comparison.png'))
plt.close()

# Step 12: Save and convert model
cnn.save(os.path.join(MODEL_DIR, 'my_model.h5'), include_optimizer=True)
print("Model saved as HDF5")

converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
tflite_model = converter.convert()
with open(os.path.join(MODEL_DIR, 'model.tflite'), 'wb') as file:
    file.write(tflite_model)
print("Model converted and saved as TFLite")
