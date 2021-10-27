#!/usr/bin/env python3.7.9
"""
Copyright © 2021 DUE TUL
@ desc  : This modules is used to load raw data
@ author: BOHAO CHU
"""
import tensorflow as tf
import reader
import config
import numpy as np

# make model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=20, kernel_size=3, activation=tf.nn.relu, input_shape=(config.width, config.height, 1)),
    tf.keras.layers.Conv2D(filters=50, kernel_size=3, activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])

model.summary()

# define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_dataset = reader.train_reader_tfrecord(config.data_path, 200)
# main function
if __name__ == "__main__":
    for batch_id, data in enumerate(train_dataset):
        datas = data['data'].numpy().reshape((-1, config.width, config.height, 1))
        labels = data['label']

        with tf.GradientTape() as tape:
            predictions = model(datas)
            # get loss of train
            train_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            train_loss = tf.reduce_mean(train_loss)
            # get accuracy of train
            train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, predictions)
            train_accuracy = np.sum(train_accuracy.numpy()) / len(train_accuracy.numpy())
            # update gradients
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if batch_id % 10 == 0:
            print("Batch %d, Loss %f, Accuracy %f" % (batch_id, train_loss.numpy(), train_accuracy))
            # save model
        if batch_id % 100 ==0:
            model.save(filepath='models/mpu.h5')