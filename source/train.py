#!/usr/bin/env python3.7.9
"""
Copyright © 2021 DUE TUL
@ desc  : This modules is used to load raw data
@ author: BOHAO CHU
"""
import config
import reader
import torch
import torch.nn as nn


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
        self.hidden = nn.Linear(32 * 32 * 17, 100)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        output = self.sigmoid(x)
        return output

class Mpu(nn.Module):
    def __init__(self):
        super(Mpu, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.hidden = nn.Linear(32 * 32 * 1, 100)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        output = self.sigmoid(x)
        return output


class Svm(nn.Module):
    def __init__(self):
        super(Svm, self).__init__()
        self.hidden = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio, mpu):
        x = torch.cat((audio, mpu),dim=1)
        x = self.hidden(x)
        out = self.sigmoid(x)
        return out


class Sensor(nn.Module):
    def __init__(self):
        super(Sensor, self).__init__()
        self.audio = Audio()
        self.mpu = Mpu()
        self.svm = Svm()


    def forward(self, x):
        in_audio = x[:, :, :, 0:71]
        in_mpu = x[:, :, :, 71:77]
        out_audio = self.audio(in_audio)
        out_mpu = self.mpu(in_mpu)
        out_svm = self.svm(out_audio, out_mpu)
        return out_audio, out_mpu, out_svm


class SENAI(nn.Module):
    def __init__(self):
        super(SENAI, self).__init__()
        self.model = Sensor()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.loss_function = nn.BCELoss()
        self.train_dataset = reader.train_reader_tfrecord("data/tfrecords/train.tfrecord", num_epochs=200, width=128, height=77)
        self.test_dataset = reader.test_reader_tfrecord("data/tfrecords/train.tfrecord")

    def train(self):
        for batch_id, data in enumerate(self.train_dataset):
            datas = torch.tensor(data['data'].numpy().reshape((-1, 1, 128, 77)))
            labels = torch.tensor(data['label'].numpy()).view(-1, 1).float()
            audio_out, mpu_out, svm_out = self.model(datas)
            svm_loss = self.loss_function(svm_out, labels)
            mpu_loss = self.loss_function(audio_out, labels)
            audio_loss = self.loss_function(mpu_out, labels)
            self.loss = svm_loss + mpu_loss + audio_loss

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            if batch_id % 10 == 0:
                svm_correct = (svm_out.ge(0.5) == labels).sum().item()
                svm_accuracy = svm_correct / labels.shape[0]
                print('Epoch:', batch_id, "Loss:%.5f" % self.loss, "Accuracy:", svm_accuracy)

            if batch_id % 100 == 0:
                svm_losses = list()
                svm_accuracies = list()
                audio_losses = list()
                audio_accuracies = list()
                mpu_losses = list()
                mpu_accuracies = list()
                for batch_id, data in enumerate(self.test_dataset):
                    datas = torch.tensor(data['data'].numpy().reshape((-1, 1, 128, 77)))
                    labels = torch.tensor(data['label'].numpy()).view(-1, 1).float()
                    audio_out, mpu_out, svm_out = self.model(datas)

                    svm_loss = self.loss_function(svm_out, labels)
                    svm_correct = (svm_out.ge(0.5) == labels).sum().item()
                    svm_accuracy = svm_correct / labels.shape[0]
                    svm_losses.append(svm_loss)
                    svm_accuracies.append(svm_accuracy)

                    audio_loss = self.loss_function(audio_out, labels)
                    audio_correct = (audio_out.ge(0.5) == labels).sum().item()
                    audio_accuracy = audio_correct / labels.shape[0]
                    audio_losses.append(audio_loss)
                    audio_accuracies.append(audio_accuracy)

                    mpu_loss = self.loss_function(mpu_out, labels)
                    mpu_correct = (mpu_out.ge(0.5) == labels).sum().item()
                    mpu_accuracy = mpu_correct / labels.shape[0]
                    mpu_losses.append(mpu_loss)
                    mpu_accuracies.append(mpu_accuracy)
                print('=================================================')
                print("SVM, Loss %f, Accuracy %f" % (
                    sum(svm_losses) / len(svm_losses), sum(svm_accuracies) / len(svm_accuracies)))
                print("MIC, Loss %f, Accuracy %f" % (
                    sum(audio_losses) / len(audio_losses), sum(audio_accuracies) / len(audio_accuracies)))
                print("MPU, Loss %f, Accuracy %f" % (
                    sum(mpu_losses) / len(mpu_losses), sum(mpu_accuracies) / len(mpu_accuracies)))
                print('=================================================')
                torch.save(self.model, "models/sensor.pkl")
                #print(self.model.state_dict().keys())
                print(self.model.state_dict()['svm.hidden.weight'],self.model.state_dict()['svm.hidden.bias'])



# main function
if __name__ == "__main__":
    # load data
    model = SENAI()
    model.train()


