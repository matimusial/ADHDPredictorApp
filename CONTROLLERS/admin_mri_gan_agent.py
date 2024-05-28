from PyQt5 import uic
import subprocess
import sys
import MRI.config
import os

current_dir = os.path.dirname(__file__)
UI_PATH = rf'{current_dir}/UI'
parent_directory = os.path.dirname(current_dir)


class AdminMriGan():
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.ui = uic.loadUi(rf'{parent_directory}/UI/aUI_projekt_GAN.ui', mainWindow)

        self.ui.startButton.clicked.connect(self.train_gan)

    def EEG_CNN_OPEN(self):
        script = rf'{parent_directory}/CONTROLLERS/admin_eeg_cnn_agent.py'
        subprocess.run(['python', script])

    def MRI_CNN_OPEN(self):
        script = rf'{parent_directory}/CONTROLLERS/admin_mri_cnn_agent.py'
        subprocess.run(['python', script])

    def train_gan(self):
        if self.ui.textEdit_epochs.toPlainText().strip() == "":
            epochs = MRI.config.GAN_EPOCHS_MRI
        else:
            epochs = int(self.ui.textEdit_epochs.toPlainText())

        if self.ui.textEdit_batch_size.toPlainText().strip() == "":
            batch_size = MRI.config.GAN_BATCH_SIZE_MRI
        else:
            batch_size = int(self.ui.textEdit_batch_size.toPlainText())

        if self.ui.textEdit_learning_rate.toPlainText().strip() == "":
            learning_rate = MRI.config.GAN_LEARNING_RATE
        else:
            learning_rate = float(self.ui.textEdit_learning_rate.toPlainText())

        if self.ui.textEdit_input_size.toPlainText().strip() == "":
            input_shape = MRI.config.GAN_INPUT_SHAPE_MRI
        else:
            input_shape = int(self.ui.textEdit_input_size.toPlainText())

        MRI.config.set_gan_epochs(epochs)
        MRI.config.set_gan_batch_size(batch_size)
        MRI.config.set_gan_learning_rate(learning_rate)
        MRI.config.set_gan_input_shape(input_shape)

        print("GAN_EPOCHS:", MRI.config.GAN_EPOCHS_MRI)
        print("GAN_BATCH_SIZE:", MRI.config.GAN_BATCH_SIZE_MRI)
        print("GAN_LEARNING_RATE:", MRI.config.GAN_LEARNING_RATE)
        print("GAN_TEST_SIZE:", MRI.config.GAN_INPUT_SHAPE_MRI)

        #odpal uczenie mri gan
