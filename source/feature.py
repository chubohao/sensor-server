#!/usr/bin/env python3.7.9
"""
Copyright © 2021 DUE TUL
@ desc  : This modules is used to load raw data
@ author: BOHAO CHU
"""
import os
import scipy.signal
import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import scale


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# add data to TFRecord
def data_example(data, label):
    feature = {
        'data': _float_feature(data),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_mpu_tfrecord(mpu_list_path, train_save_path):
    with open(mpu_list_path, 'r+') as file_mpu:
        mpu_paths = []
        mpu_labels = []
        mpu_data = file_mpu.readlines()
        print("read mpu foler")
        for single_mpu in mpu_data:
            mpu_path, mpu_label = single_mpu.replace('\n', '').split('\t')
            time, x, y, z = np.loadtxt(mpu_path, delimiter=',', unpack=True, dtype=np.float64)
            for i in range(int(len(x) / 1000)):
                index = int(1000 * i)
                data_x = x[index:(index + 1000)]
                mpu_paths.append(data_x)
                mpu_labels.append(mpu_label)
        with tf.io.TFRecordWriter(train_save_path) as train:
            for i in tqdm(range(len(mpu_paths))):
                f, t, ps = scipy.signal.stft(mpu_paths[i], fs=1000, nperseg=256, noverlap=128, boundary=None,padded=None)
                ps = ps[1:, :]
                ps = np.abs(ps)
                a = np.min(ps)
                ps[0:1] = [a, a, a, a, a, a]
                # (128, 6)
                mpu_feature = scale(ps)
                tf_example = data_example(mpu_feature.reshape(-1).tolist(), int(mpu_labels[i]))
                train.write(tf_example.SerializeToString())


def create_audio_tfrecord(audio_data_list_path, train_save_path):
    with open(audio_data_list_path, 'r+') as file_audio:
        audio_paths = []
        audio_labels = []

        audio_data = file_audio.readlines()
        print("read audio foler")
        for single_audio in audio_data:
            audio_path, audio_label = single_audio.replace('\n', '').split('\t')
            audio_paths.append(audio_path)
            audio_labels.append(audio_label)


    with tf.io.TFRecordWriter(train_save_path) as train:
        for i in tqdm(range(len(audio_paths))):
            audio_data, sr = librosa.load(audio_paths[i], sr=16000, dtype=np.float64)
            audio_data = audio_data * 2
            f, t, ps = scipy.signal.stft(audio_data, fs=16000, nperseg=256, noverlap=32, boundary=None, padded=None)
            ps = ps[1:, :]
            ps = np.abs(ps)
            # (128, 71)
            audio_feature = scale(ps)
            tf_example = data_example(audio_feature.reshape(-1).tolist(), int(audio_labels[i]))
            train.write(tf_example.SerializeToString())


def create_feature_tfrecord(audio_list_path, mpu_list_path, train_save_path, test_save_path):
    audio_paths = []
    audio_labels = []

    mpu_paths = []
    mpu_labels = []
    with open(audio_list_path, 'r+') as audio_folder, open(mpu_list_path, 'r+') as mpu_folder:
        audio_data = audio_folder.readlines()
        print("read audio foler")
        for single_audio in audio_data:
            audio_path, audio_label = single_audio.replace('\n', '').split('\t')
            audio_paths.append(audio_path)
            audio_labels.append(int(audio_label))

        mpu_data = mpu_folder.readlines()
        print("read mpu foler")
        for single_mpu in mpu_data:
            mpu_path, mpu_label = single_mpu.replace('\n', '').split('\t')
            time, x, y, z = np.loadtxt(mpu_path, delimiter=',', unpack=True, dtype=np.float64)
            for i in range(int(len(x) / 1000)):
                index = int(1000 * i)
                data_x = x[index:(index + 1000)]
                mpu_paths.append(data_x)
                mpu_labels.append(int(mpu_label))
        print(len(audio_paths), len(mpu_paths), sum(audio_labels), sum(mpu_labels))

    with tf.io.TFRecordWriter(train_save_path) as train, tf.io.TFRecordWriter(test_save_path) as test:
        for i in tqdm(range(len(audio_paths))):
            if audio_labels[i] != mpu_labels[i]:
                print("Error")
                break
            audio_data, sr = librosa.load(audio_paths[i], sr=16000, dtype=np.float64)
            audio_data = audio_data * 2
            f, t, ps = scipy.signal.stft(audio_data, fs=16000, nperseg=256, noverlap=32, boundary=None, padded=None)
            ps = ps[1:, :]
            ps = np.abs(ps)
            audio_feature = scale(ps)

            f, t, ps = scipy.signal.stft(mpu_paths[i], fs=1000, nperseg=256, noverlap=128, boundary=None, padded=None)
            ps = ps[1:, :]
            ps = np.abs(ps)
            a = np.min(ps)
            ps[0:1] = [a, a, a, a, a, a]
            mpu_feature = scale(ps)

            merge_feature = np.column_stack((audio_feature, mpu_feature))
            merge_feature = merge_feature.reshape(-1).tolist()
            tf_example = data_example(merge_feature, int(mpu_labels[i]))
            if i % 7 == 0:
                test.write(tf_example.SerializeToString())
            else:
                train.write(tf_example.SerializeToString())


# Generate audio data list
def get_audio_data_list(audio_path, list_path):
    audio_class_dir = os.listdir(audio_path)
    with open(list_path, 'w') as f_audio:
        for i in range(len(audio_class_dir)):
            sound_dir = os.listdir(os.path.join(audio_path, audio_class_dir[i]))
            for sound_file in sound_dir:
                sound_file_path = os.path.join(audio_path, audio_class_dir[i], sound_file)
                f_audio.write('%s\t%d\n' % (sound_file_path, i))
            print("audio：%d/%d  %d" % (i + 1, len(audio_class_dir), len(sound_dir)))


# Generate mpu data list
def get_mpu_data_list(mpu_path, list_path):
    mpu_class_dir = os.listdir(mpu_path)
    with open(list_path, 'w') as f_mpu:
        for i in range(len(mpu_class_dir)):
            mpu_dir = os.listdir(os.path.join(mpu_path, mpu_class_dir[i]))
            for mpu_file in mpu_dir:
                mpu_file_path = os.path.join(mpu_path, mpu_class_dir[i], mpu_file)
                f_mpu.write('%s\t%d\n' % (mpu_file_path, i))
            print("mpu  ：%d/%d  %d" % (i + 1, len(mpu_class_dir), len(mpu_dir)))



# main function
if __name__ == "__main__":
    get_audio_data_list('data/rawdata/audio', 'data/lists/audio_data_list.txt')
    get_mpu_data_list('data/rawdata/mpu', 'data/lists/mpu_data_list.txt')
    #create_audio_tfrecord('data/lists/audio_data_list.txt', 'data/tfrecords/audio_train.tfrecord')
    #create_mpu_tfrecord('data/lists/mpu_data_list.txt','data/tfrecords/mpu_train.tfrecord')
    create_feature_tfrecord('data/lists/audio_data_list.txt', 'data/lists/mpu_data_list.txt','data/tfrecords/train.tfrecord','data/tfrecords/test.tfrecord')
