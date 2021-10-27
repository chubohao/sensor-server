#!/usr/bin/env python3.7.9
'''
Copyright © 2021 DUE TUL
@ desc  : This modules is used to define the data read function
@ author: BOHAO CHU
'''
import tensorflow as tf
global_width = 128
global_height = 77

def _parse_data_function(example):
    global global_width
    global global_height
    data_feature_description = {
        'data': tf.io.FixedLenFeature([global_width * global_height], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, data_feature_description)

def train_reader_tfrecord(data_path, num_epochs, width=128, height=77, batch_Size=64):
    global global_width
    global_width = width
    global global_height
    global_height = height
    raw_dataset = tf.data.TFRecordDataset(data_path)
    train_dataset = raw_dataset.map(_parse_data_function)      # 解析
    train_dataset = train_dataset.repeat(count=num_epochs)     # 复制
    train_dataset = train_dataset.shuffle(buffer_size=1000)    # 打乱
    train_dataset = train_dataset.batch(batch_size=batch_Size) # 分批
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset


def test_reader_tfrecord(data_path, width=128, height=77, batch_size=64):
    global global_width
    global_width = width
    global global_height
    global_height = height
    raw_dataset = tf.data.TFRecordDataset(data_path)
    test_dataset = raw_dataset.map(_parse_data_function)
    test_dataset = test_dataset.shuffle(buffer_size=1000)  # 打乱
    test_dataset = test_dataset.batch(batch_size=batch_size)
    return test_dataset
