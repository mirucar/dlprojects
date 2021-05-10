import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=False, vertical_flip=False, rescale=None,
    preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_iterator = train_datagen.flow_from_directory(
        'augmented-data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='sparse')
validation_iterator = test_datagen.flow_from_directory(
        'augmented-data/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='sparse')

model = Sequential()

model.add(layers.Input(shape=train_iterator.image_shape))
model.add(layers.Conv2D(4,3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3,3), strides=3))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(3,activation='softmax'))


stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

model.compile(
  loss='sparse_categorical_crossentropy', 
  optimizer=keras.optimizers.Adam(learning_rate=0.01), 
  metrics=['accuracy'])

history = model.fit(
  train_iterator,
  epochs=100,
  steps_per_epoch=20,
  validation_data=validation_iterator,
  callbacks=[stop])

# Do Matplotlib extension below
# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')
 
# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping
fig.tight_layout()
 
fig.savefig('static/images/my_plots.png')

