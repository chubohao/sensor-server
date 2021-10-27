#!/usr/bin/env python3.7.9
"""
Copyright © 2021 DUE TUL
@ desc  : This modules is used to load raw data
@ author: BOHAO CHU
"""
import config
import reader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.functional import F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import svm
import joblib
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


class Svm(nn.Module):
    def __init__(self):
        super(Svm, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        pre = self.linear(x)
        return pre


# main function
if __name__ == "__main__":
    # load data
    loss_function = nn.CrossEntropyLoss()  # 实例化损失函数
    dataset = reader.test_reader_tfrecord("data/tfrecords/test.tfrecord", width=128, height=77)

    audio_model = torch.load("models/audio.pkl")
    test_audio_loss = list()
    test_audio_accuracy = list()

    mpu_model = torch.load("models/mpu.pkl")
    test_mpu_loss = list()
    test_mpu_accuracy = list()
    new_result = torch.tensor([])
    for batch_id, data in enumerate(dataset):
        datas = torch.tensor(data['data'].numpy().reshape((-1, 1, 128, 77)))
        labels = torch.tensor(data['label'].numpy())
        audio_out = audio_model(datas[:, :, :, 0:71])
        audio_loss = loss_function(audio_out, labels)
        test_audio_loss.append(audio_loss)
        audio_pred = torch.max(audio_out, 1)[1].data.squeeze()
        audio_accuracy = sum(audio_pred == labels) / labels.size(0)
        test_audio_accuracy.append(audio_accuracy)

        mpu_out = mpu_model(datas[:, :, :, 71:77])
        mpu_loss = loss_function(mpu_out, labels)
        test_mpu_loss.append(mpu_loss)
        mpu_pred = torch.max(mpu_out, 1)[1].data.squeeze()
        mpu_accuracy = sum(mpu_pred == labels) / labels.size(0)
        test_mpu_accuracy.append(mpu_accuracy)
        result = torch.stack([audio_pred, mpu_pred, labels])
        new_result = torch.cat((new_result,result),1)
    print(new_result[:, 1:50])


    print('=================================================')
    print("Test, Loss %f, Accuracy %f" % (
        sum(test_audio_loss) / len(test_audio_loss), sum(test_audio_accuracy) / len(test_audio_accuracy)))
    print('=================================================')

    print('=================================================')
    print("Test, Loss %f, Accuracy %f" % (
        sum(test_mpu_loss) / len(test_mpu_loss), sum(test_mpu_accuracy) / len(test_mpu_accuracy)))
    print('=================================================')

    clf = svm.SVC(kernel='linear')
    x = new_result[0:2,:].t() + 0.1*torch.normal(0, 1, (400, 2))
    y = new_result[2:,:].t()
    yy = (new_result[0:2,:].t()[:,0:1] - y)*0.5 + y
    plt.scatter(x[:,0], x[:,1],c=yy)
    clf.fit(x, y)
    w = clf.coef_[0]  # 获取w
    a = -w[0] / w[1]  # 斜率
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)
    print(w)
    plt.show()
    joblib.dump(clf, "svm.m")
    print(clf.intercept_[0])
    '''
    model = Mpu()  # 实例化模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 实例化优化器
    loss_function = nn.CrossEntropyLoss()  # 实例化损失函数
    train_dataset = reader.train_reader_tfrecord("data/tfrecords/train.tfrecord", num_epochs=50, width=128, height=77)
    for batch_id, data in enumerate(train_dataset):
        datas = torch.tensor(data['data'].numpy().reshape((-1, 1, 128, 77)))
        labels = torch.tensor(data['label'].numpy())
        output = model(datas[:,:,:,71:77])
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 10 == 0:
            pred = torch.max(output, 1)[1].data.squeeze()
            accuracy = sum(pred == labels) / labels.size(0)
            print('Epoch:', batch_id, "Loss:%.5f" %loss.item(), "Accuracy:", accuracy.item())
            torch.save(model, "models/mpu.pkl")
    '''






