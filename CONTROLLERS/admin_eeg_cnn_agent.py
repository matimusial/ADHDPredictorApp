from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QFileDialog
import EEG.config
from EEG.TRAIN.train import train_cnn_eeg
from CONTROLLERS.DBConnector import DBConnector
import os
import shutil

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
UI_PATH = os.path.join(current_dir, 'UI')
parent_directory = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(parent_dir, 'EEG', 'temp_model_path')
TRAIN_PATH = os.path.join(parent_dir, 'EEG', 'TRAIN', 'TRAIN_DATA')
PREDICT_PATH = os.path.join(parent_dir, 'EEG', 'PREDICT', 'PREDICT_DATA')

class AdminEegCnn:
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.ui = uic.loadUi(os.path.join(parent_directory, 'UI', 'aUI_projekt_EEG.ui'), mainWindow)

        self.pathTrain = TRAIN_PATH
        self.db_conn = None

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

        self.thread = QThread()

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
        train_cnn_eeg(True, self.pathTrain, PREDICT_PATH, MODEL_PATH, self.ui)

    def onFinished(self):
        self.connect_to_db()
        file_name = os.listdir(MODEL_PATH)
        file_path = os.path.join('./EEG/temp_model_path', file_name[0])
        print(file_name[0])
        self.db_conn.insert_data_into_models_table(
            file_name[0], file_path, EEG.config.EEG_NUM_OF_ELECTRODES, EEG.config.CNN_INPUT_SHAPE, 'cnn_eeg', EEG.config.FS, None, "eeg_cnn_model")
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

    def onError(self, error):
        print(f"Error: {error}")

    def connect_to_db(self):
        self.db_conn = DBConnector()
        print(self.db_conn.connection)
        if self.db_conn.connection is None: return

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