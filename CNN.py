from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt


inc_model=InceptionV3(include_top=False,
                      weights='imagenet',
                      input_shape=((3, 150, 150)))

bottleneck_datagen = ImageDataGenerator(rescale=1. / 255)  # собственно, генератор

train_generator = bottleneck_datagen.flow_from_directory('data/img_train/',
                                                         target_size=(150, 150),
                                                         batch_size=32,
                                                         class_mode=None,
                                                         shuffle=False)

validation_generator = bottleneck_datagen.flow_from_directory('data/img_val/',
                                                              target_size=(150, 150),
                                                              batch_size=32,
                                                              class_mode=None,
                                                              shuffle=False)

bottleneck_features_train = inc_model.predict_generator(train_generator, 2000)
np.save(open('bottleneck_features/bn_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = inc_model.predict_generator(validation_generator, 2000)
np.save(open('bottleneck_features/bn_features_validation.npy', 'wb'), bottleneck_features_validation)

train_data = np.load(open('bottleneck_features_and_weights/bn_features_train.npy', 'rb'))
train_labels = np.array([0] * 1000 + [1] * 1000)

validation_data = np.load(open('bottleneck_features_and_weights/bn_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 1000 + [1] * 1000)

fc_model = Sequential()
fc_model.add(Flatten(input_shape=train_data.shape[1:]))
fc_model.add(Dense(64, activation='relu', name='dense_one'))
fc_model.add(Dropout(0.5, name='dropout_one'))
fc_model.add(Dense(64, activation='relu', name='dense_two'))
fc_model.add(Dropout(0.5, name='dropout_two'))
fc_model.add(Dense(1, activation='sigmoid', name='output'))

fc_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

fc_model.fit(train_data, train_labels,
            nb_epoch=50, batch_size=32,
            validation_data=(validation_data, validation_labels))

fc_model.save_weights('bottleneck_features_and_weights/fc_inception_cats_dogs_250.hdf5') # сохраняем веса

fc_model.evaluate(validation_data, validation_labels)

weights_filename='bottleneck_features_and_weights/fc_inception_cats_dogs_250.hdf5'

x = Flatten()(inc_model.output)
x = Dense(64, activation='relu', name='dense_one')(x)
x = Dropout(0.5, name='dropout_one')(x)
x = Dense(64, activation='relu', name='dense_two')(x)
x = Dropout(0.5, name='dropout_two')(x)
top_model=Dense(1, activation='sigmoid', name='output')(x)
model = Model(input=inc_model.input, output=top_model)
weights_filename='bottleneck_features_and_weights/fc_inception_cats_dogs_250.hdf5'
model.load_weights(weights_filename, by_name=True)


for layer in inc_model.layers[:205]:
    layer.trainable = False


model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
                #optimizer='rmsprop',
              metrics=['accuracy'])

filepath="new_model_weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/img_train/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/img_val/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')


pred_generator=test_datagen.flow_from_directory('data/img_val/',
                                                     target_size=(150,150),
                                                     batch_size=100,
                                                     class_mode='binary')


model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=200,
        validation_data=validation_generator,
        nb_val_samples=2000,
    callbacks=callbacks_list)