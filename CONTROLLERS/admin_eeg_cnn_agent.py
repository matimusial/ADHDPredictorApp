from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog
import EEG.config
from EEG.TRAIN.train import *
import numpy as np
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
UI_PATH = rf'{current_dir}/UI'
parent_directory = os.path.dirname(current_dir)

MODEL_PATH = rf'{parent_dir}/EEG/CNN/MODEL'
TRAIN_PATH = rf'{parent_dir}/EEG/TRAIN/TRAIN_DATA'
PREDICT_PATH = rf'{parent_dir}/EEG/PREDICT/PREDICT_DATA'

class AdminEegCnn:
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.ui = uic.loadUi(rf'{parent_directory}/UI/aUI_projekt_EEG.ui', mainWindow)

        self.pathTrain = TRAIN_PATH

        self.ui.folder_explore.clicked.connect(self.showDialog)
        self.ui.startButton.clicked.connect(self.train_cnn)

    def showDialog(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')

        if folder:
            self.pathTrain = folder
            self.ui.path_label.setText(f'{folder}')

    def train_cnn(self):
        if self.ui.textEdit_epochs.toPlainText().strip() == "":
            epochs = EEG.config.CNN_EPOCHS
        else:
            epochs = int(self.ui.textEdit_epochs.toPlainText())

        if self.ui.textEdit_batch_size.toPlainText().strip() == "":
            batch_size = EEG.config.CNN_BATCH_SIZE
        else:
            batch_size = int(self.ui.textEdit_batch_size.toPlainText())

        if self.ui.textEdit_learning_rate.toPlainText().strip() == "":
            learning_rate = EEG.config.CNN_LEARNING_RATE
        else:
            learning_rate = int(self.ui.textEdit_learning_rate.toPlainText())

        if self.ui.textEdit_electrodes.toPlainText().strip() == "":
            electrodes = EEG.config.EEG_NUM_OF_ELECTRODES
        else:
            electrodes = int(self.ui.textEdit_electrodes.toPlainText())

        if self.ui.textEdit_frame_size.toPlainText().strip() == "":
            frame_size = EEG.config.EEG_SIGNAL_FRAME_SIZE
        else:
            frame_size = int(self.ui.textEdit_frame_size.toPlainText())

        if self.ui.textEdit_frequency.toPlainText().strip() == "":
            frequency = EEG.config.FS
        else:
            frequency = int(self.ui.textEdit_frequency.toPlainText())

        EEG.config.set_cnn_epochs(epochs)
        EEG.config.set_cnn_batch_size(batch_size)
        EEG.config.set_learning_rate(learning_rate)
        EEG.config.set_electrodes(electrodes)
        EEG.config.set_frame(frame_size)
        EEG.config.set_fs(frequency)

        print("CNN_EPOCHS:", EEG.config.CNN_EPOCHS)
        print("CNN_BATCH_SIZE:", EEG.config.CNN_BATCH_SIZE)
        print("CNN_LEARNING_RATE:", EEG.config.CNN_LEARNING_RATE)
        print("EEG_NUM_OF_ELECTRODES:", EEG.config.EEG_NUM_OF_ELECTRODES)
        print("EEG_SIGNAL_FRAME_SIZE:", EEG.config.EEG_SIGNAL_FRAME_SIZE)
        print("FS:", EEG.config.FS)

        train_cnn_eeg(True, TRAIN_PATH, PREDICT_PATH, MODEL_PATH)
