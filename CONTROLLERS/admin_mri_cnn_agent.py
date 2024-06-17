import sys

from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QSize
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QProgressBar, QPushButton

import MRI.config
from MRI.CNN.train import train_cnn, readPickleForUI
from CONTROLLERS.DBConnector import DBConnector
import os
import shutil

from CONTROLLERS.metrics import RealTimeMetrics

class AdminMriCnn:
    def __init__(self, mainWindow, ui_path, main_path):
        self.MAIN_PATH = main_path
        self.mainWindow = mainWindow
        self.MODEL_PATH = os.path.join(self.MAIN_PATH, 'MRI', 'CNN', 'temp_model_path')
        self.TRAIN_PATH = os.path.join(self.MAIN_PATH, 'MRI', 'REAL_MRI')
        self.PREDICT_PATH = os.path.join(self.MAIN_PATH, 'MRI', 'CNN', 'PREDICT_DATA')
        self.ui = uic.loadUi(os.path.join(ui_path, 'aUI_projekt_MRI.ui'), mainWindow)
        self.modelTrained = False
        adhd_data, control_data = readPickleForUI(self.TRAIN_PATH)

        self.loaded_adhd_files = len(adhd_data)
        self.loaded_control_files = len(control_data)
        self.currChannels = len(adhd_data[0])

        self.updateInfoDump()

        self.pathTrain = self.TRAIN_PATH
        self.db_conn = None

        self.model_description = ""
        self.ui.status_label_2.setText("STATUS: Await")
        self.ui.db_status_2.setText("STATUS: Await")

        self.ui.textEdit_epochs_2.setValue(MRI.config.CNN_EPOCHS_MRI)
        self.ui.textEdit_batch_size_2.setValue(MRI.config.CNN_BATCH_SIZE_MRI)
        self.ui.textEdit_learning_rate_2.setValue(MRI.config.CNN_LEARNING_RATE_MRI)
        self.ui.textEdit_test_size.setValue(MRI.config.TEST_SIZE_MRI_CNN)

        self.ui.startButton_2.clicked.connect(self.train_mri)
        self.ui.stopButton_2.clicked.connect(self.stopModel)
        self.ui.exitButton.clicked.connect(self.on_exit)
        self.ui.save_db_2.clicked.connect(self.sendToDb)
        self.ui.del_model_2.clicked.connect(self.delModel)

        self.run_stop_controller = False

        self.progressBar = self.ui.findChild(QProgressBar, "progressBar_2")

    def updateInfoDump(self):
        self.ui.info_dump_2.setText(
            f'{self.loaded_adhd_files + self.loaded_control_files} files in dir (ADHD: {self.loaded_adhd_files}; CONTROL: {self.loaded_control_files})\n'
            f'{self.currChannels} channels'
        )

    def train_mri(self):
        self.current_size = self.ui.size()
        self.ui.setFixedSize(self.current_size)
        self.ui.status_label_2.setText("STATUS: Starting")
        MRI.CNN.train.modelStopFlag = False
        self.run_stop_controller = True

        epochs = self.ui.textEdit_epochs_2.value()
        batch_size = self.ui.textEdit_batch_size_2.value()
        learning_rate = self.ui.textEdit_learning_rate_2.value()
        test_size = self.ui.textEdit_test_size.value()

        self.model_description = self.ui.model_description_2.toPlainText()

        if epochs is False or batch_size is False or learning_rate is False:
            self.invalid_input_msgbox()
        else:
            MRI.config.CNN_EPOCHS_MRI = epochs
            MRI.config.CNN_BATCH_SIZE_MRI = batch_size
            MRI.config.CNN_LEARNING_RATE_MRI = learning_rate
            MRI.config.CNN_TEST_SIZE_MRI_CNN = test_size

            self.thread = QThread()

            self.toggle_buttons(False)
            self.progressBar.setValue(0)
            self.real_time_metrics = RealTimeMetrics(epochs, self.progressBar, self.ui.plotLabel_CNN_2)
            self.real_time_metrics.start()

            # Create a worker object
            self.worker = Worker(self)

            # Move the worker to the thread
            self.worker.moveToThread(self.thread)

            # Connect signals and slots
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.onFinished)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.error.connect(self.onError)

            # Start the thread
            self.thread.start()

    def train(self):
        if os.path.exists(self.MODEL_PATH):
            shutil.rmtree(self.MODEL_PATH)
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)
        self.ui.status_label_2.setText("STATUS: Running")
        out = train_cnn(True, self.pathTrain, self.PREDICT_PATH, self.MODEL_PATH)
        if out == "STOP":
            self.ui.status_label_2.setText("STATUS: Await")
            MRI.CNN.train.modelStopFlag = False

    def onFinished(self):
        file_name = os.listdir(self.MODEL_PATH)
        if file_name:
            acc = file_name[0].replace(".keras", "")
            self.ui.more_info_dump_2.setText(f"Final model accuracy: {acc}")
        else:
            self.ui.more_info_dump_2.setText("Warning: Could not find model file in MODEL_PATH")

        self.ui.status_label_2.setText("STATUS: Model done")
        self.ui.setFixedSize(QSize(16777215, 16777215))
        self.toggle_buttons(True)
        self.modelTrained = True

    def toggle_buttons(self, state):
        try:
            self.ui.CNN_EEG_Button.setEnabled(state)
            self.ui.GAN_MRI_Button.setEnabled(state)
            self.ui.dbButton_2.setEnabled(state)
            self.ui.switchSceneBtn.setEnabled(state)
            self.ui.startButton_2.setEnabled(state)
            self.ui.save_db_2.setEnabled(state)
            self.ui.del_model_2.setEnabled(state)
        except Exception as e:
            print(f'Failed toggle_buttons: {e}')

    def sendToDb(self):
        if not self.modelTrained: return
        self.modelTrained = False
        if os.path.exists(self.MODEL_PATH):
            file_name = os.listdir(self.MODEL_PATH)
            self.ui.db_status_2.setText("STATUS: Connecting...")
            self.connect_to_db()
            self.ui.db_status_2.setText("STATUS: Sending...")
            file_path = os.path.join(self.MAIN_PATH, 'MRI', 'CNN', 'temp_model_path', file_name[0])
            self.ui.status_label_2.setText("STATUS: Uploading model")
            self.db_conn.insert_data_into_models_table(
                file_name[0].replace(".keras", ""), file_path, None,
                MRI.config.CNN_INPUT_SHAPE_MRI, 'cnn_mri', None, None,
                f"learning rate: {MRI.config.CNN_LEARNING_RATE_MRI}; batch size: {MRI.config.CNN_BATCH_SIZE_MRI}; epochs: {MRI.config.CNN_EPOCHS_MRI}; {self.model_description}"
            )
            self.ui.status_label_2.setText("STATUS: Await")
            self.ui.db_status_2.setText("STATUS: Await")
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
                        self.ui.status_label.setText("STATUS: Await")
                        self.delete_done_msgbox()
                        print(f"Plik {file_name} został usunięty.")
                    except Exception as e:
                        print(f"Nie można usunąć pliku {file_name}: {e}")
                else:
                    print(f"{file_name} nie jest plikiem.")
            else:
                print("Katalog jest pusty, nie ma plików do usunięcia.")

        else:
            print("Nie ma ścieżki MODEL_PATH")

    def onError(self, error):
        print(f"Error: {error}")
        self.toggle_buttons(True)

    def on_exit(self):
        QApplication.quit()

    def stopModel(self):
        if self.run_stop_controller:
            MRI.CNN.train.modelStopFlag = True
            self.real_time_metrics.stop()
            self.ui.status_label_2.setText("STATUS: Stopping...")

    def connect_to_db(self):
        self.db_conn = DBConnector()
        self.db_conn.establish_connection()
        if self.db_conn.connection is None: return


    def invalid_input_msgbox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Invalid input.")
        msg.setWindowTitle("Error")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def upload_done_msgbox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Data successfully sent to database.")
        msg.setWindowTitle("Operation successfully")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def delete_done_msgbox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Data successfully deleted.")
        msg.setWindowTitle("Operation successfully")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def run(self):
        try:
            self.train_cnn_eeg()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def train_cnn_eeg(self):
        self.controller.train()
