import sys
from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QSize, Qt
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QSpinBox, QProgressBar
from PyQt5.QtGui import QFontMetrics

import MRI.config
from MRI.GAN.train import train_gan
from MRI.CNN.train import readPickleForUI
from CONTROLLERS.DBConnector import DBConnector
import os
import shutil

import pickle
from PIL import Image
import numpy as np

from CONTROLLERS.metrics import RealTimeMetrics_GEN

class AdminMriGan:
    def __init__(self, mainWindow, ui_path, main_path):
        self.MAIN_PATH = main_path
        self.mainWindow = mainWindow
        self.ui = uic.loadUi(os.path.join(ui_path, 'aUI_projekt_GAN.ui'), mainWindow)
        self.MODEL_PATH = os.path.join(self.MAIN_PATH, 'MRI', 'GAN', 'temp_model_path')
        self.TRAIN_PATH = os.path.join(self.MAIN_PATH, 'MRI', 'REAL_MRI')
        self.IMAGES_PATH = os.path.join(self.MAIN_PATH, 'MRI', 'Img_to_pkl')
        self.PREDICT_PATH = os.path.join(self.MAIN_PATH, 'MRI', 'GAN', 'MODELS')
        adhd_data, control_data = readPickleForUI(self.TRAIN_PATH)

        self.loaded_adhd_files = len(adhd_data)
        self.loaded_control_files = len(control_data)
        self.currChannels = len(adhd_data[0])

        self.updateInfoDump()

        self.pathTrain = self.TRAIN_PATH
        self.db_conn = None
        self.modelTrained = False
        self.checked = "ADHD"

        self.model_description = ""
        self.ui.status_label.setText("STATUS: Await")
        self.ui.db_status.setText("STATUS: Await")

        self.ui.textEdit_epochs.setValue(MRI.config.GAN_EPOCHS_MRI)
        self.ui.textEdit_batch_size.setValue(MRI.config.GAN_BATCH_SIZE_MRI)
        self.ui.textEdit_learning_rate.setValue(MRI.config.GAN_LEARNING_RATE)
        self.ui.textEdit_test_size.setValue(MRI.config.TEST_SIZE_MRI_GAN)
        self.ui.textEdit_print_interval.setValue(MRI.config.TRAIN_GAN_PRINT_INTERVAL)
        self.ui.textEdit_disp_interval.setValue(MRI.config.TRAIN_GAN_DISP_INTERVAL)

        self.ui.folder_explore.clicked.connect(self.showDialog_1)
        self.ui.folder_explore_2.clicked.connect(self.showDialog_2)
        self.ui.folder_explore_3.clicked.connect(self.showDialog_3)
        self.ui.startButton.clicked.connect(self.train_gan)
        self.ui.exitButton_2.clicked.connect(self.on_exit)
        self.ui.stopButton.clicked.connect(self.stopModel)
        self.ui.save_db.clicked.connect(self.sendToDb)

        self.run_stop_controller = False

        self.progressBar = self.ui.findChild(QProgressBar, "progressBar")
        self.gan_generation_warning_msgbox()

        self.delModel()

    def showDialog_1(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')

        new_size = (128, 120)

        # List to store resized images
        resized_images = []

        # Loop through all files in the directory
        for filename in os.listdir(self.IMAGES_PATH):
            if filename.endswith('.png'):
                # Load the image
                image_path = os.path.join(self.IMAGES_PATH, filename)
                image = Image.open(image_path)

                # Resize the image
                resized_image = image.resize(new_size)

                # Convert the image to a numpy array
                image_array = np.array(resized_image)

                # Append the numpy array to the list
                resized_images.append(image_array)

                # Remove the original image file
                os.remove(image_path)

        pickle_filename = os.path.join(folder, 'ADHD_REAL.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(resized_images, f)

    def showDialog_2(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')

        new_size = (128, 120)

        # List to store resized images
        resized_images = []

        # Loop through all files in the directory
        for filename in os.listdir(self.IMAGES_PATH):
            if filename.endswith('.png'):
                # Load the image
                image_path = os.path.join(self.IMAGES_PATH, filename)
                image = Image.open(image_path)

                # Resize the image
                resized_image = image.resize(new_size)

                # Convert the image to a numpy array
                image_array = np.array(resized_image)

                # Append the numpy array to the list
                resized_images.append(image_array)

                # Remove the original image file
                os.remove(image_path)

        pickle_filename = os.path.join(folder, 'CONTROL_REAL.pkl')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(resized_images, f)

    def showDialog_3(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')
        try:
            self.TRAIN_PATH = folder
            adhd_data, control_data = readPickleForUI(self.TRAIN_PATH)
            print(adhd_data)
            print(control_data)
            self.loaded_adhd_files = len(adhd_data)
            self.loaded_control_files = len(control_data)
            self.currChannels = len(adhd_data[0])
            self.updateInfoDump()
        except Exception as e:
            print(f'Failed {self.MODEL_PATH}. Reason: {e}')
        self.pathTrain = self.TRAIN_PATH



    def updateInfoDump(self):
        self.ui.info_dump.setText(
            f'{self.loaded_adhd_files + self.loaded_control_files} files in dir (ADHD: {self.loaded_adhd_files}; CONTROL: {self.loaded_control_files})\n'
            f'{self.currChannels} channels'
        )

    def train_gan(self):
        self.delModel()
        self.current_size = self.ui.size()
        self.ui.setFixedSize(self.current_size)
        self.ui.status_label.setText("STATUS: Starting")
        MRI.GAN.train.modelStopFlag = False
        self.run_stop_controller = True

        epochs = self.ui.textEdit_epochs.value()
        batch_size = self.ui.textEdit_batch_size.value()
        learning_rate = self.ui.textEdit_learning_rate.value()
        test_size = self.ui.textEdit_test_size.value()
        print_interval = self.ui.textEdit_print_interval.value()
        disp_interval = self.ui.textEdit_disp_interval.value()

        self.model_description = self.ui.model_description.toPlainText()

        self.progressBar = self.ui.findChild(QProgressBar, "progressBar")

        if epochs is False or batch_size is False or learning_rate is False:
            self.invalid_input_msgbox()
        else:
            MRI.config.GAN_EPOCHS_MRI = epochs
            MRI.config.GAN_BATCH_SIZE_MRI = batch_size
            MRI.config.GAN_LEARNING_RATE = learning_rate
            MRI.config.CNN_TEST_SIZE_MRI_GAN = test_size
            MRI.config.TRAIN_GAN_PRINT_INTERVAL = print_interval
            MRI.config.TRAIN_GAN_DISP_INTERVAL = disp_interval

            self.toggle_buttons(False)
            self.progressBar.setValue(0)
            self.thread = QThread()
            self.real_time_metrics = RealTimeMetrics_GEN(epochs, self.progressBar, MRI.config.TRAIN_GAN_PRINT_INTERVAL, MRI.config.TRAIN_GAN_DISP_INTERVAL, self.ui.plotLabel_GAN_plot, self.ui.plotLabel_GAN_image)
            self.real_time_metrics.start()

            self.worker = Worker(self)

            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.onFinished)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.error.connect(self.onError)

            self.thread.start()

    def train(self):
        if os.path.exists(self.MODEL_PATH):
            shutil.rmtree(self.MODEL_PATH)
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)
        if self.ui.radioButton_Control.isChecked():
            train_gan(True, "CONTROL", self.pathTrain, self.PREDICT_PATH)
        elif self.ui.radioButton_ADHD.isChecked():
            train_gan(True, "ADHD", self.pathTrain, self.PREDICT_PATH)
        else:
            self.invalid_input_msgbox()

    def onFinished(self):
        file_name = os.listdir(self.MODEL_PATH)
        if file_name:
            acc = file_name[0].replace(".keras", "")
            self.ui.more_info_dump.setText(f"Final model accuracy: {acc}")
        else:
            self.ui.more_info_dump.setText("Warning: Could not find model file in MODEL_PATH")

        self.ui.status_label.setText("STATUS: Model done")
        self.ui.setFixedSize(QSize(16777215, 16777215))
        self.toggle_buttons(True)
        self.modelTrained = True

    def toggle_buttons(self, state):
        try:
            self.ui.CNN_EEG_Button.setEnabled(state)
            self.ui.CNN_MRI_Button.setEnabled(state)
            self.ui.dbButton.setEnabled(state)
            self.ui.switchSceneBtn.setEnabled(state)
            self.ui.startButton.setEnabled(state)
            self.ui.save_db.setEnabled(state)
        except Exception as e:
            print(f'Failed toggle_buttons: {e}')

    def sendToDb(self):
        if not self.modelTrained: return
        self.modelTrained = False

        if os.path.exists(self.MODEL_PATH):
            file_name = os.listdir(self.MODEL_PATH)
            self.ui.db_status.setText("STATUS: Connecting...")
            self.connect_to_db()
            self.ui.db_status.setText("STATUS: Sending...")
            file_path = os.path.join(self.MAIN_PATH, 'MRI', 'GAN', 'temp_model_path', file_name[0])
            self.ui.status_label.setText("STATUS: Uploading model")
            if self.checked == "ADHD":
                self.db_conn.insert_data_into_models_table(
                    file_name[0].replace(".keras", ""), file_path, None,
                    MRI.config.GAN_INPUT_SHAPE_MRI, 'gan_adhd', None, "A",
                    f"learning rate: {MRI.config.GAN_LEARNING_RATE}; batch size: {MRI.config.GAN_BATCH_SIZE_MRI}; epochs: {MRI.config.GAN_EPOCHS_MRI}; {self.model_description}"
                )
            else:
                self.db_conn.insert_data_into_models_table(
                    file_name[0].replace(".keras", ""), file_path, None,
                    MRI.config.GAN_INPUT_SHAPE_MRI, 'gan_control', None, "A",
                    f"learning rate: {MRI.config.GAN_LEARNING_RATE}; batch size: {MRI.config.GAN_BATCH_SIZE_MRI}; epochs: {MRI.config.GAN_EPOCHS_MRI}; {self.model_description}"
                )
            self.ui.status_label.setText("STATUS: Await")
            self.ui.db_status.setText("STATUS: Await")
            self.upload_done_msgbox()
            for filename in os.listdir(self.MODEL_PATH):
                file_path = os.path.join(self.MODEL_PATH, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

            try:
                os.rmdir(self.MODEL_PATH)
            except Exception as e:
                print(f'Failed to delete the directory {self.MODEL_PATH}. Reason: {e}')
        else:
            print("No model to upload")

    def delModel(self):
        if os.path.exists(self.MODEL_PATH):
            file_list = os.listdir(self.MODEL_PATH)
            if file_list:
                file_name = file_list[0]
                file_path = os.path.join(self.MODEL_PATH, file_name)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        self.ui.status_label.setText("STATUS: Model deleted")
                    except Exception as e:
                        print(f"Error deleting file: {e}")
                        self.ui.status_label.setText("STATUS: Error deleting model")
                else:
                    print("File not found")
            else:
                print("Directory is empty")
        else:
            print("Model directory does not exist")

    def gan_generation_warning_msgbox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("WARNING")
        msg.setText("Generating synthetic data using GAN models can be a time-consuming process, especially for large datasets. Please ensure your system has enough resources and patience for the process to complete.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def invalid_input_msgbox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Invalid input. Please check your parameters.")
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def upload_done_msgbox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Model uploaded successfully.")
        msg.setWindowTitle("Success")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def model_upload_failed(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Model upload has failed.")
        msg.setWindowTitle("Error")
        msg.exec_()

    def connect_to_db(self):
        try:
            self.db_conn = DBConnector('localhost', 'admin', 'test', 'postgres', 'password')
        except Exception as e:
            print(f"Failed to connect to the database: {e}")

    def on_exit(self):
        MRI.GAN.train.modelStopFlag = True
        QApplication.quit()

    def stopModel(self):
        if self.run_stop_controller:
            self.real_time_metrics.stop()
            self.ui.status_label.setText("STATUS: Stopping...")

    def onError(self, error):
        print(f"Error occurred: {error}")
        self.toggle_buttons(True)

class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, admin_mri_gan):
        super().__init__()
        self.admin_mri_gan = admin_mri_gan

    def run(self):
        try:
            self.admin_mri_gan.train()
        except Exception as e:
            self.error.emit(str(e))
        self.finished.emit()
