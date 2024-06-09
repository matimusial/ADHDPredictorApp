import sys

from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QProgressBar

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
        self.TRAIN_PATH = os.path.join(self.MAIN_PATH, 'INPUT_DATA', 'EEG', 'MAT')
        self.PREDICT_PATH = os.path.join(self.MAIN_PATH, 'EEG', 'PREDICT', 'PREDICT_DATA')
        self.ui = uic.loadUi(os.path.join(ui_path, 'aUI_projekt_EEG.ui'), mainWindow)

        #_, _, initChannels, adhdcount, controlcount = read_eeg_raw(self.TRAIN_PATH)

        self.loaded_adhd_files = 0
        self.loaded_control_files = 0
        self.currChannels = 0

        self.updateInfoDump()

        self.pathTrain = self.TRAIN_PATH
        self.db_conn = None

        self.model_description = ""
        self.ui.status_label.setText("STATUS: Await")
        self.ui.db_status.setText("STATUS: Await")

        self.ui.textEdit_epochs.setPlainText(str(EEG.config.CNN_EPOCHS))
        self.ui.textEdit_batch_size.setPlainText(str(EEG.config.CNN_BATCH_SIZE))
        self.ui.textEdit_learning_rate.setPlainText(str(EEG.config.CNN_LEARNING_RATE))
        self.ui.textEdit_electrodes.setPlainText(str(self.currChannels))
        self.ui.textEdit_frame_size.setPlainText(str(EEG.config.EEG_SIGNAL_FRAME_SIZE))
        self.ui.textEdit_frequency.setPlainText(str(EEG.config.FS))
        #self.ui.path_label.setText(f'{self.TRAIN_PATH}')
        #self.ui.path_label.setTextElideMode(Qt.ElideRight)

        self.ui.textEdit_frequency.setReadOnly(True)
        self.ui.textEdit_electrodes.setReadOnly(True)
        self.ui.textEdit_frame_size.setReadOnly(True)

        self.ui.folder_explore.clicked.connect(self.showDialog)
        self.ui.startButton.clicked.connect(self.train_cnn)
        self.ui.stopButton.clicked.connect(self.stopModel)
        self.ui.exitButton.clicked.connect(self.on_exit)
        self.ui.save_db.clicked.connect(self.sendToDb)
        self.ui.del_model.clicked.connect(self.delModel)

        self.run_stop_controller = False

        self.progressBar = self.ui.findChild(QProgressBar, "progressBar")

    def showDialog(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')

        if folder:
            adhd_path = os.path.join(folder, 'ADHD')
            control_path = os.path.join(folder, 'CONTROL')

            if os.path.isdir(adhd_path) and os.path.isdir(control_path):
                self.pathTrain = folder
                self.ui.path_label.setText(f'{folder}')
                _, _, initChannels, adhdcount, controlcount = read_eeg_raw(self.TRAIN_PATH)
                self.loaded_adhd_files = adhdcount
                self.loaded_control_files = controlcount
                self.currChannels = initChannels[0]['shape'][0]
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
            f'{self.currChannels} channels'
        )
        self.ui.textEdit_electrodes.setPlainText(str(self.currChannels))

    def train_cnn(self):
        self.ui.status_label.setText("STATUS: Starting")
        EEG.TRAIN.train.modelStopFlag = False
        self.run_stop_controller = True

        epochs = self.validate_epochs()
        batch_size = self.validate_batch_size()
        frame_size = self.validate_frame_size()
        learning_rate = self.validate_learning_rate()

        self.model_description = self.ui.model_description.toPlainText()

        if self.ui.textEdit_electrodes.toPlainText().strip() == "":
            electrodes = EEG.config.EEG_NUM_OF_ELECTRODES
        else:
            electrodes = int(self.ui.textEdit_electrodes.toPlainText())

        if self.ui.textEdit_frequency.toPlainText().strip() == "":
            frequency = EEG.config.FS
        else:
            frequency = int(self.ui.textEdit_frequency.toPlainText())

        if epochs is False or batch_size is False or learning_rate is False:
            self.invalid_input_msgbox()
        else:
            EEG.config.CNN_EPOCHS = epochs
            EEG.config.CNN_BATCH_SIZE = batch_size
            EEG.config.CNN_LEARNING_RATE = learning_rate
            EEG.config.EEG_NUM_OF_ELECTRODES = electrodes
            EEG.config.EEG_SIGNAL_FRAME_SIZE = frame_size
            EEG.config.FS = frequency

            print("CNN_EPOCHS:", EEG.config.CNN_EPOCHS)
            print("CNN_BATCH_SIZE:", EEG.config.CNN_BATCH_SIZE)
            print("CNN_LEARNING_RATE:", EEG.config.CNN_LEARNING_RATE)
            print("EEG_NUM_OF_ELECTRODES:", EEG.config.EEG_NUM_OF_ELECTRODES)
            print("EEG_SIGNAL_FRAME_SIZE:", EEG.config.EEG_SIGNAL_FRAME_SIZE)
            print("FS:", EEG.config.FS)

            self.thread = QThread()

            # Reset the plot and clear metrics before starting the training
            self.progressBar.setValue(0)
            self.real_time_metrics = RealTimeMetrics(epochs, self.progressBar, self.ui.plotLabel_CNN)
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
        self.ui.status_label.setText("STATUS: Running")
        out = train_cnn_eeg_readraw(True, self.pathTrain, self.PREDICT_PATH, self.MODEL_PATH)
        if out == "STOP":
            self.ui.status_label.setText("STATUS: Await")
            EEG.TRAIN.train.modelStopFlag = False

    def onFinished(self):
        file_name = os.listdir(self.MODEL_PATH)
        acc = file_name[0].replace(".keras", "")
        self.ui.more_info_dump.setText(f"Final model accuracy: {acc}")
        self.ui.status_label.setText("STATUS: Model done")
    def sendToDb(self):
        if os.path.exists(self.MODEL_PATH):
            file_name = os.listdir(self.MODEL_PATH)
            self.ui.db_status.setText("STATUS: Connecting...")
            self.connect_to_db()
            self.ui.db_status.setText("STATUS: Sending...")
            file_path = os.path.join(self.MAIN_PATH, 'EEG', 'temp_model_path', file_name[0])
            self.ui.status_label.setText("STATUS: Uploading model")
            self.db_conn.insert_data_into_models_table(
                file_name[0].replace(".keras", ""), file_path, EEG.config.EEG_NUM_OF_ELECTRODES,
                EEG.config.CNN_INPUT_SHAPE, 'cnn_eeg', EEG.config.FS, None,
                f"learning rate: {EEG.config.CNN_LEARNING_RATE}; batch size: {EEG.config.CNN_BATCH_SIZE}; epochs: {EEG.config.CNN_EPOCHS}; {self.model_description}"
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
        print(self.db_conn.connection)
        if self.db_conn.connection is None: return

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
            if value is None or value <= 0 or value >= 1 or not isinstance(value, float):
                print(f"WARNING: '{text}' is invalid.\nLearning rate value must be a float between 0 and 1 (exclusive).\n")
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

