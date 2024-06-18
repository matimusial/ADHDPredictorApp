import sys

from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt, QSize
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QProgressBar
from PyQt5.QtGui import QFontMetrics

import EEG.config
from EEG.TRAIN.train import train_cnn_eeg_readraw
from CONTROLLERS.DBConnector import DBConnector
from CONTROLLERS.file_io import read_eeg_raw
import os
import shutil

from CONTROLLERS.metrics import RealTimeMetrics

class AdminEegCnn:
    def __init__(self, mainWindow, ui_path, main_path):
        self.MAIN_PATH = main_path
        self.mainWindow = mainWindow
        self.MODEL_PATH = os.path.join(self.MAIN_PATH, 'EEG', 'temp_model_path')
        self.PREDICT_PATH = os.path.join(self.MAIN_PATH, 'EEG', 'PREDICT', 'PREDICT_DATA')
        self.ui = uic.loadUi(os.path.join(ui_path, 'aUI_projekt_EEG.ui'), mainWindow)

        self.loaded_adhd_files = 0
        self.loaded_control_files = 0
        self.currChannels = 0
        self.currSamples = 0

        self.updateInfoDump()

        self.pathTrain = None
        self.db_conn = None

        self.model_description = ""
        self.ui.status_label.setText("STATUS: Await")
        self.ui.db_status.setText("STATUS: Await")

        self.ui.textEdit_epochs.setValue(EEG.config.CNN_EPOCHS)
        self.ui.textEdit_batch_size.setValue(EEG.config.CNN_BATCH_SIZE)
        self.ui.textEdit_learning_rate.setValue(EEG.config.CNN_LEARNING_RATE)
        self.ui.textEdit_test_size.setValue(EEG.config.TEST_SIZE_EEG_CNN)
        self.ui.textEdit_frame_size.setValue(EEG.config.EEG_SIGNAL_FRAME_SIZE)

        self.ui.textEdit_electrodes.setPlainText(str(self.currChannels))
        self.ui.textEdit_frequency.setPlainText(str(EEG.config.FS))

        self.ui.textEdit_frequency.setReadOnly(True)
        self.ui.textEdit_electrodes.setReadOnly(True)

        self.ui.folder_explore.clicked.connect(self.showDialog)
        self.ui.startButton.clicked.connect(self.train_cnn)
        self.ui.stopButton.clicked.connect(self.stopModel)
        self.ui.exitButton.clicked.connect(self.on_exit)
        self.ui.save_db.clicked.connect(self.sendToDb)

        frame_size = self.ui.textEdit_frame_size.value()
        EEG.config.EEG_SIGNAL_FRAME_SIZE = frame_size

        self.run_stop_controller = False

        self.progressBar = self.ui.findChild(QProgressBar, "progressBar")

        self.delModel()

    def showDialog(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')

        if folder:
            adhd_path = os.path.join(folder, 'ADHD')
            control_path = os.path.join(folder, 'CONTROL')

            if os.path.isdir(adhd_path) and os.path.isdir(control_path):
                self.pathTrain = folder
                metrics = QFontMetrics(self.ui.path_label.font())
                elided_text = metrics.elidedText(folder, Qt.ElideMiddle, self.ui.path_label.width())
                self.ui.path_label.setText(elided_text)
                _, _, initChannels, adhdcount, controlcount = read_eeg_raw(folder)
                self.loaded_adhd_files = adhdcount
                self.loaded_control_files = controlcount
                self.currChannels = initChannels[0]['shape'][0]
                self.currSamples = initChannels[0]['shape'][1]
                self.updateInfoDump()
            else:
                self.invalid_folder_msgbox()

    def contains_files(self, folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.mat') or file.endswith('.csv') or file.endswith('.edf'):
                return True
        return False

    def updateInfoDump(self):
        self.ui.info_dump.setText(
            f'{self.loaded_adhd_files + self.loaded_control_files} files in dir (ADHD: {self.loaded_adhd_files}; CONTROL: {self.loaded_control_files})\n'
            f'{self.currChannels} channels; {self.currSamples} samples'
        )
        self.ui.textEdit_electrodes.setPlainText(str(self.currChannels))

    def train_cnn(self):
        self.delModel()
        self.current_size = self.ui.size()
        self.ui.setFixedSize(self.current_size)
        self.ui.status_label.setText("STATUS: Starting")
        EEG.TRAIN.train.modelStopFlag = False
        self.run_stop_controller = True

        epochs = self.ui.textEdit_epochs.value()
        batch_size = self.ui.textEdit_batch_size.value()
        learning_rate = self.ui.textEdit_learning_rate.value()

        test_size = self.ui.textEdit_test_size.value()
        self.model_description = self.ui.model_description.toPlainText()

        if epochs is False or batch_size is False or learning_rate is False:
            self.invalid_input_msgbox()
        else:
            EEG.config.CNN_EPOCHS = epochs
            EEG.config.CNN_BATCH_SIZE = batch_size
            EEG.config.CNN_LEARNING_RATE = learning_rate
            EEG.config.CNN_TEST_SIZE = test_size

            self.thread = QThread()

            self.toggle_buttons(False)
            self.progressBar.setValue(0)
            self.real_time_metrics = RealTimeMetrics(epochs, self.progressBar, self.ui.plotLabel_CNN)
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
        if not os.path.exists(self.MODEL_PATH):
            try:
                os.makedirs(self.MODEL_PATH)
            except Exception as e:
                print(f"err: {e}")

        self.ui.status_label.setText("STATUS: Running")
        if self.pathTrain.endswith("ADHD") or self.pathTrain.endswith("CONTROL"):
            self.pathTrain = self.pathTrain[:-5]
        out = train_cnn_eeg_readraw(True, self.pathTrain, self.PREDICT_PATH, self.MODEL_PATH)
        if out == "STOP":
            self.ui.status_label.setText("STATUS: Await")
            EEG.TRAIN.train.modelStopFlag = False

    def onFinished(self):
        file_name = os.listdir(self.MODEL_PATH)
        if file_name:
            acc = file_name[0].replace(".keras", "")
            self.ui.more_info_dump.setText(f"Final model accuracy: {acc}")
        else:
            self.ui.more_info_dump.setText(f"Warning: Could not find model file in MODEL_PATH")

        self.ui.status_label.setText("STATUS: Model done")
        self.ui.setFixedSize(QSize(16777215, 16777215))
        self.toggle_buttons(True)

    def toggle_buttons(self,state):
        try:
            self.ui.CNN_MRI_Button.setEnabled(state)
            self.ui.GAN_MRI_Button.setEnabled(state)
            self.ui.dbButton.setEnabled(state)
            self.ui.switchSceneBtn.setEnabled(state)
            self.ui.folder_explore.setEnabled(state)
            self.ui.startButton.setEnabled(state)
            self.ui.save_db.setEnabled(state)
        except Exception as e:
            print(f'Failed toggle_buttons: {e}')

    def sendToDb(self):
        if os.path.exists(self.MODEL_PATH):
            if any(os.path.isfile(os.path.join(self.MODEL_PATH, f)) for f in os.listdir(self.MODEL_PATH)):
                file_name = os.listdir(self.MODEL_PATH)
            else:
                self.model_upload_failed()
                return
            self.ui.db_status.setText("STATUS: Connecting...")
            conn = self.connect_to_db()
            if conn:
                self.ui.db_status.setText("STATUS: Sending...")
                file_path = os.path.join(self.MAIN_PATH, 'EEG', 'temp_model_path', file_name[0])
                self.ui.status_label.setText("STATUS: Uploading model")
                try:
                    self.db_conn.insert_data_into_models_table(
                        file_name[0].replace(".keras", ""), file_path, EEG.config.EEG_NUM_OF_ELECTRODES,
                        EEG.config.CNN_INPUT_SHAPE, 'cnn_eeg', EEG.config.FS, None,
                        f"learning rate: {EEG.config.CNN_LEARNING_RATE}; batch size: {EEG.config.CNN_BATCH_SIZE}; epochs: {EEG.config.CNN_EPOCHS}; {self.model_description}"
                    )
                    self.upload_done_msgbox()
                except Exception as e:
                    print(f'Failed to upload model to db. Reason: {e}')
                    self.model_upload_failed()
                self.ui.status_label.setText("STATUS: Await")
                self.ui.db_status.setText("STATUS: Await")
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
                print(f'Database connection error')
                self.model_upload_failed()
        else:
            print("No model to upload")

    def delModel(self):
        if os.path.exists(self.MODEL_PATH):
            file_list = os.listdir(self.MODEL_PATH)
            if file_list:
                for file_name in file_list:
                    file_path = os.path.join(self.MODEL_PATH, file_name)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            print(f"Plik {file_name} został usunięty.")
                        except Exception as e:
                            print(f"Nie można usunąć pliku {file_name}: {e}")
                    else:
                        print(f"{file_name} nie jest plikiem.")
                self.ui.status_label.setText("STATUS: Await")
            else:
                print("Katalog jest pusty, nie ma plików do usunięcia.")
        else:
            print("Nie ma ścieżki MODEL_PATH")

    def onError(self, error):
        self.toggle_buttons(True)
        print(f"Error: {error}")

    def on_exit(self):
        QApplication.quit()

    def stopModel(self):
        if self.run_stop_controller:
            EEG.TRAIN.train.modelStopFlag = True
            self.real_time_metrics.stop()
            self.ui.status_label.setText("STATUS: Stopping...")

    def connect_to_db(self):
        self.db_conn = DBConnector()
        self.db_conn.establish_connection()
        if self.db_conn.connection is None:
            return False
        else:
            return True

    def validate_input(self, text):
        try:
            num = float(text)
            if num.is_integer():
                return int(num)
            else:
                return num
        except ValueError:
            return None

    def validate_epochs(self):
        text = self.ui.textEdit_epochs.toPlainText().strip()
        if text == "":
            print(f"WARNING: Field is empty.\n")
            return False
        else:
            value = self.validate_input(text)
            if value is None or value <= 1 or not isinstance(value, int):
                print(f"WARNING: '{text}' is invalid.\nEpochs value must be an integer greater than 1.\n")
                return False
            else:
                return value

    def validate_batch_size(self):
        text = self.ui.textEdit_batch_size.toPlainText().strip()
        if text == "":
            print(f"WARNING: Field is empty.\n")
            return False
        else:
            value = self.validate_input(text)
            if value is None or value <= 1 or not isinstance(value, int):
                print(f"WARNING: '{text}' is invalid.\nBatch size value must be an integer greater than 1.\n")
                return False
            else:
                return value

    def validate_frame_size(self):
        text = self.ui.textEdit_frame_size.toPlainText().strip()
        if text == "":
            print(f"WARNING: Field is empty.\n")
            return False
        else:
            value = self.validate_input(text)
            if value is None or value <= 1 or not isinstance(value, int):
                print(f"WARNING: '{text}' is invalid.\nFrame size value must be an integer greater than 1.\n")
                return False
            else:
                return value

    def validate_learning_rate(self):
        text = self.ui.textEdit_learning_rate.toPlainText().strip()
        if text == "":
            print(f"WARNING: Field is empty.\n")
            return False
        else:
            value = self.validate_input(text)
            if value is None or value <= 0.0001 or value >= 1 or not isinstance(value, float):
                print(f"WARNING: '{text}' is invalid.\nLearning rate value must be a float between 0.0001 and 1 (exclusive).\n")
                return False
            else:
                return value

    def validate_test_size(self):
        text = self.ui.textEdit_test_size.toPlainText().strip()
        if text == "":
            print(f"WARNING: Field is empty.\n")
            return False
        else:
            value = self.validate_input(text)
            if value is None or value <= 0.1 or value >= 0.9 or not isinstance(value, float):
                print(f"WARNING: '{text}' is invalid.\nTest size value must be a float between 0.1 and 0.9 (exclusive).\n")
                return False
            else:
                return value

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

    def invalid_folder_msgbox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Invalid input folder")
        msg.setWindowTitle("Error")
        msg.exec_()

    def model_upload_failed(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Model upload has failed.")
        msg.setWindowTitle("Error")
        msg.exec_()

    def change_btn_state(self, state):
        self.ui.CNN_MRI_Button.setEnabled(state)
        self.ui.GAN_MRI_Button.setEnabled(state)
        self.ui.startButton.setEnabled(state)
        self.ui.dbButton.setEnabled(state)
        self.ui.switchSceneBtn.setEnabled(state)
        self.ui.folder_explore.setEnabled(state)


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

