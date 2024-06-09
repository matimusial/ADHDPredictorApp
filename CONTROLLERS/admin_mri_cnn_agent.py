from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QProgressBar

import MRI.config
from MRI.CNN.train import train_cnn
from CONTROLLERS.DBConnector import DBConnector
import os
import shutil

from CONTROLLERS.metrics import RealTimeMetrics



current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
UI_PATH = os.path.join(current_dir, 'UI')
parent_directory = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(parent_dir, 'MRI', 'CNN', 'temp_model_path')
TRAIN_PATH = os.path.join(parent_dir, 'MRI', 'REAL_MRI')
PREDICT_PATH = os.path.join(parent_dir, 'MRI', 'CNN', 'PREDICT_DATA')

class AdminMriCnn:
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.ui = uic.loadUi(os.path.join(parent_directory, 'UI', 'aUI_projekt_MRI.ui'), mainWindow)

        self.loaded_adhd_files = 0
        self.loaded_control_files = 0
        self.currChannels = 0

        self.updateInfoDump()

        self.pathTrain = TRAIN_PATH
        self.db_conn = None

        self.model_description = ""
        self.ui.status_label_2.setText("STATUS: Await")
        self.ui.db_status_2.setText("STATUS: Await")

        self.ui.textEdit_epochs_2.setPlainText(str(MRI.config.CNN_EPOCHS_MRI))
        self.ui.textEdit_batch_size_2.setPlainText(str(MRI.config.CNN_BATCH_SIZE_MRI))
        self.ui.textEdit_learning_rate_2.setPlainText(str(MRI.config.CNN_LEARNING_RATE_MRI))
        self.ui.textEdit_electrodes_2.setPlainText(str(self.currChannels))
        self.ui.textEdit_frame_size_2.setPlainText(str(MRI.config.CNN_SINGLE_INPUT_SHAPE_MRI))
        self.ui.path_label_2.setText(f'{TRAIN_PATH}')

        self.ui.textEdit_electrodes_2.setReadOnly(True)

        self.ui.folder_explore_2.clicked.connect(self.showDialog)
        self.ui.startButton_2.clicked.connect(self.train_mri)
        self.ui.stopButton_2.clicked.connect(self.stopModel)
        self.ui.exitButton.clicked.connect(self.on_exit)
        self.ui.save_db_2.clicked.connect(self.sendToDb)
        self.ui.del_model_2.clicked.connect(self.delModel)

        self.progressBar = self.ui.findChild(QProgressBar, "progressBar_2")

    def showDialog(self):
        folder = QFileDialog.getExistingDirectory(self.ui, 'Wybierz folder')

        if folder:
            self.pathTrain = folder
            self.ui.path_label.setText(f'{folder}')
            self.loaded_adhd_files = 0
            self.loaded_control_files = 0
            self.currChannels = 0
            self.updateInfoDump()

    def updateInfoDump(self):
        self.ui.info_dump_2.setText(
            f'{self.loaded_adhd_files + self.loaded_control_files} files in dir (ADHD: {self.loaded_adhd_files}; CONTROL: {self.loaded_control_files})\n'
            f'{self.currChannels} channels'
        )
        self.ui.textEdit_electrodes_2.setPlainText(str(self.currChannels))

    def train_mri(self):
        self.ui.status_label_2.setText("STATUS: Starting")
        MRI.CNN.train.modelStopFlag = False

        epochs = self.validate_epochs()
        batch_size = self.validate_batch_size()
        frame_size = self.validate_frame_size()
        learning_rate = self.validate_learning_rate()

        self.model_description = self.ui.model_description_2.toPlainText()

        if self.ui.textEdit_electrodes_2.toPlainText().strip() == "":
            electrodes = 120
        else:
            electrodes = int(self.ui.textEdit_electrodes_2.toPlainText())

        if self.ui.textEdit_frame_size_2.toPlainText().strip() == "":
            input_shape = MRI.config.CNN_SINGLE_INPUT_SHAPE_MRI
        else:
            input_shape = int(self.ui.textEdit_frame_size_2.toPlainText())

        if epochs is False or batch_size is False or learning_rate is False:
            self.invalid_input_msgbox()
        else:
            MRI.config.CNN_EPOCHS_MRI = epochs
            MRI.config.CNN_BATCH_SIZE_MRI = batch_size
            MRI.config.CNN_LEARNING_RATE_MRI = learning_rate
            MRI.config.CNN_SINGLE_INPUT_SHAPE_MRI = input_shape

            print("MRI_EPOCHS:", MRI.config.CNN_EPOCHS_MRI)
            print("MRI_BATCH_SIZE:", MRI.config.CNN_BATCH_SIZE_MRI)
            print("MRI_LEARNING_RATE:", MRI.config.CNN_LEARNING_RATE_MRI)
            print("MRI_FRAME_SIZE:", MRI.config.CNN_SINGLE_INPUT_SHAPE_MRI)

            self.thread = QThread()

            # Reset the plot and clear metrics before starting the training
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
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        self.ui.status_label_2.setText("STATUS: Running")
        out = train_cnn(True, self.pathTrain, PREDICT_PATH, MODEL_PATH)
        if out == "STOP":
            self.ui.status_label_2.setText("STATUS: Await")
            MRI.CNN.train.modelStopFlag = False

    def onFinished(self):
        file_name = os.listdir(MODEL_PATH)
        acc = file_name[0].replace(".keras", "")
        self.ui.more_info_dump_2.setText(f"Final model accuracy: {acc}")
        self.ui.status_label_2.setText("STATUS: Model done")

    def sendToDb(self):
        file_name = os.listdir(MODEL_PATH)
        if file_name:
            self.ui.db_status_2.setText("STATUS: Connecting...")
            self.connect_to_db()
            self.ui.db_status_2.setText("STATUS: Sending...")
            file_path = os.path.join('./MRI/CNN/temp_model_path', file_name[0])
            self.ui.status_label_2.setText("STATUS: Uploading model")
            self.db_conn.insert_data_into_models_table(
                file_name[0].replace(".keras", ""), file_path, None,
                MRI.config.CNN_INPUT_SHAPE_MRI, 'cnn_mri', None, None,
                f"learning rate: {MRI.config.CNN_LEARNING_RATE_MRI}; batch size: {MRI.config.CNN_BATCH_SIZE_MRI}; epochs: {MRI.config.CNN_EPOCHS_MRI}; {self.model_description}"
            )
            self.ui.status_label_2.setText("STATUS: Await")
            self.ui.db_status_2.setText("STATUS: Await")
            self.upload_done_msgbox()
            for filename in os.listdir(MODEL_PATH):
                file_path = os.path.join(MODEL_PATH, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

            try:
                os.rmdir(MODEL_PATH)
            except Exception as e:
                print(f'Failed to delete the directory {MODEL_PATH}. Reason: {e}')
        else:
            print("No model to upload")

    def delModel(self):
        if os.path.exists(MODEL_PATH):
            file_list = os.listdir(MODEL_PATH)
            if file_list:
                file_name = file_list[0]
                file_path = os.path.join(MODEL_PATH, file_name)
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
        MRI.CNN.train.modelStopFlag = True
        self.real_time_metrics.stop()
        self.ui.status_label_2.setText("STATUS: Stopping...")

    def connect_to_db(self):
        self.db_conn = DBConnector()
        self.db_conn.establish_connection()
        print(self.db_conn.connection)
        if self.db_conn.connection is None: return

    def validate_epochs(self):
        text = self.ui.textEdit_epochs_2.toPlainText().strip()
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
    def validate_input(self, text):
        try:
            num = float(text)
            if num.is_integer():
                return int(num)
            else:
                return num
        except ValueError:
            return None

    def validate_batch_size(self):
        text = self.ui.textEdit_batch_size_2.toPlainText().strip()
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
        text = self.ui.textEdit_frame_size_2.toPlainText().strip()
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
        text = self.ui.textEdit_learning_rate_2.toPlainText().strip()
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
