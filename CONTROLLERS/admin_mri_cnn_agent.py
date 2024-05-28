from PyQt5 import uic

import subprocess
import sys
import os

import MRI.config

current_dir = os.path.dirname(__file__)
UI_PATH = rf'{current_dir}/UI'
parent_directory = os.path.dirname(current_dir)


class AdminMriCnn:
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.ui = uic.loadUi(rf'{parent_directory}/UI/aUI_projekt_MRI.ui', mainWindow)

        self.ui.startButton.clicked.connect(self.train_mri)

    def EEG_CNN_OPEN(self):
        script = rf'{parent_directory}/CONTROLLERS/admin_eeg_cnn_agent.py'
        subprocess.run(['python', script])

    def MRI_GAN_OPEN(self):
        script = rf'{parent_directory}/CONTROLLERS/admin_mri_gan_agent.py'
        subprocess.run(['python', script])

    def train_mri(self):
        if self.ui.textEdit_epochs.toPlainText().strip() == "":
            epochs = MRI.config.CNN_EPOCHS_MRI
        else:
            epochs = int(self.ui.textEdit_epochs.toPlainText())

        if self.ui.textEdit_batch_size.toPlainText().strip() == "":
            batch_size = MRI.config.BATCH_SIZE_MRI
        else:
            batch_size = int(self.ui.textEdit_batch_size.toPlainText())

        if self.ui.textEdit_learning_rate.toPlainText().strip() == "":
            learning_rate = MRI.config.CNN_LEARNING_RATE_MRI
        else:
            learning_rate = int(self.ui.textEdit_learning_rate.toPlainText())

        if self.ui.textEdit_frame_size.toPlainText().strip() == "":
            input_shape = MRI.config.CNN_SINGLE_INPUT_SHAPE_MRI
        else:
            input_shape = int(self.ui.textEdit_frame_size.toPlainText())

        MRI.config.set_mri_epochs(epochs)
        MRI.config.set_mri_batch_size(batch_size)
        MRI.config.set_mri_learning_rate(learning_rate)
        MRI.config.set_mri_input_shape(input_shape)

        print("GAN_EPOCHS:", MRI.config.CNN_EPOCHS_MRI)
        print("GAN_BATCH_SIZE:", MRI.config.BATCH_SIZE_MRI)
        print("GAN_LEARNING_RATE:", MRI.config.CNN_LEARNING_RATE_MRI)
        print("GAN_TEST_SIZE:", MRI.config.CNN_SINGLE_INPUT_SHAPE_MRI)

        # Odpal uczenie się mri cnn
