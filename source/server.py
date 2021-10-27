#!/usr/bin/env python3.7.9
"""
Copyright © 2021 DUE TUL
@ date  : Tuesday May 25, 2020
@ desc  : This modules is load model and test real time data
@ author: BOHAO CHU
"""

import json
import time
import base64
import scipy
import logging
import requests
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from flask import Flask, request
from sklearn.preprocessing import scale
import scipy.signal as signal
import joblib

import config
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import svm

url = 'http://192.168.0.199:8889/'
headers = {'content-type': "application/json"}

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class Audio(nn.Module):
    def __init__(self):
        super(Audio, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 32 * 17, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class Mpu(nn.Module):
    def __init__(self):
        super(Mpu, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 32 * 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

audio_model = torch.load("models/audio.pkl")
mpu_model = torch.load("models/mpu.pkl")
svm_model = joblib.load("models/svm.m")

def predict(dicts):
    mic = dicts['mic']
    #dis_d = dicts['dis']
    #dis_f = [[dis_d] * 1 for i in range(128)]

    f, t, ps = scipy.signal.stft(mic, fs=16000, nperseg=256, noverlap=32, boundary=None, padded=None)
    ps = ps[1:, :]
    ps = np.abs(ps)
    audio_feature = scale(ps)
    audio_feature = torch.tensor(audio_feature).view(-1, 1, 128, 71).float()
    audio_out = audio_model(audio_feature)
    audio_pred = torch.max(audio_out, 1)[1]


    mpu = dicts['mpu_x']
    f, t, ps = scipy.signal.stft(mpu, fs=1000, nperseg=256, noverlap=128, boundary=None, padded=None)
    ps = ps[1:, :]
    ps = np.abs(ps)
    a = np.min(ps)
    ps[0:1] = [a, a, a, a, a, a]
    mpu_feature = scale(ps)
    mpu_feature = torch.tensor(mpu_feature).view(-1, 1, 128, 6).float()
    mpu_out = mpu_model(mpu_feature)
    mpu_pred = torch.max(mpu_out, 1)[1]
    audio_flag = None
    mpu_flag = None
    fin = torch.zeros(1,2)
    if audio_pred[0] == 1:
        fin[0,0] = 1
        audio_flag = "Rotation"
    if mpu_pred[0] == 1:
        fin[0,1] = 1
        mpu_flag = "Rotation"
    fin_flag = svm_model.predict(fin)
    if fin_flag == 1:
        fin_flag = "Rotation"
    else:
        fin_flag = "None"

    print("MIC: ", audio_flag, "MPU: ", mpu_flag, "    SVM: ", fin_flag )

    '''

    merge_f = np.column_stack((audio_tf_ps, dis_f, dis_f, dis_f, dis_f))

    data_real = merge_f.reshape((-1, 128, 75, 1))

    result = 'a'
    lab = tf.math.argmax(result, 1)
    lab = tf.keras.backend.eval(lab)
    json_r = ['{:.2f}'.format(i) for i in result[0].tolist()]

    action = ''
    if lab == 0:
        action = 'On / Bottom Moving'
    elif lab == 1:
        action = 'Off'
    elif lab == 2:
        action = 'On / Moving Left'
    elif lab == 3:
        action = 'On / Drill Rotation / Bottom Moving'
    elif lab == 4:
        action = 'On / Drill Rotation / Moving Left'
    elif lab == 5:
        action = 'On / Drill Rotation / Moving Right'
    elif lab == 6:
        action = 'On'
    elif lab == 7:
        action = 'On / Moving Right'
    elif lab == 8:
        action = 'On / Drill Rotation'

    json_result = {'prediction': action, 'result': json_r}
    '''
    return "a", "b"


def visulation(dicts, result):
    mpu_x = dicts['mpu_x']
    mpu_y = dicts['mpu_y']
    mpu_z = dicts['mpu_z']

    mpu_x = np.array(mpu_x)
    f, t, ps = signal.stft(mpu_x, fs=1000, nperseg=256, noverlap=128, boundary=None, padded=None)
    ps = ps[1:, :]
    ps = np.abs(ps)
    a = np.min(ps)
    ps[0:1] = [a, a, a, a, a, a]
    acc_x = preprocessing.scale(ps)

    mpu_y = np.array(mpu_y)
    f, t, ps = signal.stft(mpu_y, fs=1000, nperseg=256, noverlap=128, boundary=None, padded=None)
    ps = ps[1:, :]
    ps = np.abs(ps)
    a = np.min(ps)
    ps[0:1] = [a, a, a, a, a, a]
    acc_y = preprocessing.scale(ps)

    mpu_z = np.array(mpu_z)
    f, t, ps = signal.stft(mpu_z, fs=1000, nperseg=256, noverlap=128, boundary=None, padded=None)
    ps = ps[1:, :]
    ps = np.abs(ps)
    a = np.min(ps)
    ps[0:1] = [a, a, a, a, a, a]
    acc_z = preprocessing.scale(ps)


    audio = dicts['mic']
    audio_data = np.array(audio)
    f, t, ps = signal.stft(audio_data, fs=16000, nperseg=256, noverlap=32, boundary=None, padded=None)
    ps = ps[1:, :]
    mic_f = np.abs(ps)

    audio_data = mic_f

    mpux_data = np.array(acc_x).reshape(128, 6)
    mpuy_data = np.array(acc_y).reshape(128, 6)
    mpuz_data = np.array(acc_z).reshape(128, 6)

    audio_data = np.transpose(audio_data)[5::]

    tmp = []
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    for i in range(0, audio_data.shape[0], 11):
        tmp.append(np.around(np.mean(audio_data[i:i + 11], 0), decimals=2))
    tmp = np.transpose(np.array(tmp))

    for i in range(0, tmp.shape[0], 2):
        r1.append(np.mean(np.array(tmp)[i:i + 4], 0))
        r2.append(np.mean(np.array(mpux_data)[i:i + 2], 0))
        r3.append(np.mean(np.array(mpuy_data)[i:i + 2], 0))
        r4.append(np.mean(np.array(mpuz_data)[i:i + 2], 0))
    r1 = np.transpose(np.array(r1))**2
    r2 = np.transpose(np.array(r2))**2
    r3 = np.transpose(np.array(r3))**2
    r4 = np.transpose(np.array(r4))**2

    if result == "shutdown":
        r1 = r1*0.01
        r2 = r2*0.001
        r3 = r3*0.001
        r4 = r4*0.001
    elif result == "standy":
        r1 = r1*0.1
        r2 = r2*0.01
        r3 = r3*0.01
        r4 = r4*0.01
    audio = np.around(r1, decimals=2).reshape(-1).tolist()
    x = np.around(r2, decimals=2).reshape(-1).tolist()
    y = np.around(r3, decimals=2).reshape(-1).tolist()
    z = np.around(r4, decimals=2).reshape(-1).tolist()


    data = {
        'audio': audio,
        'x': x,
        'y': y,
        'z': z,
        'active': result
    }
    respose = requests.post(url, json=data)
    print(respose.text)



# communicate with phone
@app.route('/', methods=["POST"])
def hello():
    if request.method == 'POST':
        req = request.get_data()
        dicts = json.loads(req)
        result, action = predict(dicts)
        #visulation(dicts, action)
        #return action
        return "OK"

# main function
if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)

"""
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
model = tf.keras.models.load_model(r'models/final_0524_9.h5', compile=False)


def predict(dicts):
    mic = dicts['mic']
    dis_d = dicts['dis']
    dis_f = [[dis_d] * 1 for i in range(128)]

    f, t, ps = scipy.signal.stft(mic, fs=16000, nperseg=256, noverlap=32, boundary=None, padded=None)
    ps = ps[1:, :]
    ps = np.abs(ps)
    audio_tf_ps = scale(ps)

    merge_f = np.column_stack((audio_tf_ps, dis_f, dis_f, dis_f, dis_f))

    data_real = merge_f.reshape((-1, 128, 75, 1))

    result = model.predict(data_real)
    lab = tf.math.argmax(result, 1)
    lab = tf.keras.backend.eval(lab)
    json_r = ['{:.2f}'.format(i) for i in result[0].tolist()]

    action = ''
    if lab == 0:
        action = 'machine : standby bottom'
    elif lab == 1:
        action = 'machine : shutdown'
    elif lab == 2:
        action = 'machine : standby left'
    elif lab == 3:
        action = 'machine : standby rotate bottom'
    elif lab == 4:
        action = 'machine : standby rotate left'
    elif lab == 5:
        action = 'machine : standby rotate right'
    elif lab == 6:
        action = 'machine : standby'
    elif lab == 7:
        action = 'machine : standby right'
    elif lab == 8:
        action = 'machine : standby rotate'

    json_result = {'prediction': action, 'result': json_r}
    #print('prediction:', json_result)
    return json_result, action

# communicate with phone
@app.route('/', methods=["POST"])
def hello():
    if request.method == 'POST':
        req = request.get_data()
        dicts = json.loads(req)
        result, action= predict(dicts)
        return action


# main function
if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)
"""