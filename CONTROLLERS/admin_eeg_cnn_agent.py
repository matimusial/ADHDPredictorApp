from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QFileDialog
import EEG.config
from EEG.TRAIN.train import train_cnn_eeg, train_cnn_eeg_readraw
from CONTROLLERS.DBConnector import DBConnector
from CONTROLLERS.metrics import *
import os
import shutil

from PyQt5.QtCore import QMutex,QMutexLocker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtGui import QPixmap
import io
import time

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
UI_PATH = os.path.join(current_dir, 'UI')
parent_directory = os.path.dirname(current_dir)

MODEL_PATH = os.path.join(parent_dir, 'EEG', 'temp_model_path')
TRAIN_PATH = os.path.join(parent_dir, 'EEG', 'TRAIN', 'TRAIN_DATA')
PREDICT_PATH = os.path.join(parent_dir, 'EEG', 'PREDICT', 'PREDICT_DATA')

'''
TO DO:
-U̶s̶t̶a̶w̶i̶ć̶ ̶t̶y̶t̶u̶ł̶ ̶o̶k̶n̶a̶ ̶n̶a̶ ̶w̶i̶d̶o̶k̶ ̶w̶ ̶k̶t̶ó̶r̶m̶ ̶s̶i̶ę̶ ̶j̶e̶s̶t̶
-D̶o̶m̶y̶ś̶l̶n̶e̶ ̶w̶a̶r̶t̶o̶ś̶c̶i̶ ̶w̶y̶ś̶w̶i̶e̶t̶l̶o̶n̶e̶ ̶w̶ ̶p̶a̶r̶a̶m̶e̶t̶r̶a̶c̶h̶ ̶m̶o̶d̶e̶l̶u̶
-D̶o̶m̶y̶ś̶l̶n̶a̶ ̶w̶a̶r̶t̶o̶ś̶ć̶ ̶w̶y̶ś̶w̶i̶e̶t̶l̶o̶n̶a̶ ̶w̶ ̶ś̶c̶i̶e̶ż̶c̶e̶ ̶m̶o̶d̶e̶l̶u̶
-P̶o̶d̶ł̶ą̶c̶z̶y̶ć̶ ̶d̶o̶ ̶t̶r̶a̶i̶n̶'̶a̶ ̶t̶r̶a̶i̶n̶_̶c̶n̶n̶_̶e̶e̶g̶_̶r̶e̶a̶d̶r̶a̶w̶ ̶z̶a̶m̶i̶a̶s̶t̶ ̶t̶r̶a̶i̶n̶_̶c̶n̶n̶_̶e̶e̶g̶
-D̶e̶a̶k̶t̶y̶w̶o̶w̶a̶ć̶ ̶w̶a̶r̶t̶o̶ś̶c̶i̶ ̶d̶l̶a̶ ̶f̶r̶e̶q̶u̶e̶n̶c̶y̶ ̶i̶ ̶k̶a̶n̶a̶ł̶y̶
-Wyświetlenie wartości związanymi z danymi uczącymi (ilość plików, parametry itd.)
-Wspólne umiejscowienie przycisków przełączania użytkownika i admina
-dane z configa są ignorowane przez resztę kodu
-Ten graf to zadziała kiedyś?
-!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TESTOWAĆ WSZYSTKO!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
'''

class AdminEegCnn:
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.mainWindow.setWindowTitle("ADMIN: EEG for CNN")
        self.ui = uic.loadUi(os.path.join(parent_directory, 'UI', 'aUI_projekt_EEG.ui'), mainWindow)

        self.pathTrain = TRAIN_PATH
        self.db_conn = None

        self.ui.textEdit_epochs.setPlainText(str(EEG.config.CNN_EPOCHS))
        self.ui.textEdit_batch_size.setPlainText(str(EEG.config.CNN_BATCH_SIZE))
        self.ui.textEdit_learning_rate.setPlainText(str(EEG.config.CNN_LEARNING_RATE))
        self.ui.textEdit_electrodes.setPlainText(str(EEG.config.EEG_NUM_OF_ELECTRODES))
        self.ui.textEdit_frame_size.setPlainText(str(EEG.config.EEG_SIGNAL_FRAME_SIZE))
        self.ui.textEdit_frequency.setPlainText(str(EEG.config.FS))
        self.ui.path_label.setText(f'{TRAIN_PATH}')

        self.ui.textEdit_frequency.setReadOnly(True)
        self.ui.textEdit_electrodes.setReadOnly(True)
        self.ui.textEdit_frame_size.setReadOnly(True)

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
            epochs = int(self.ui.textEdit_epochs)

        if self.ui.textEdit_batch_size.toPlainText().strip() == "":
            batch_size = EEG.config.CNN_BATCH_SIZE
        else:
            batch_size = int(self.ui.textEdit_batch_size)

        if self.ui.textEdit_learning_rate.toPlainText().strip() == "":
            learning_rate = EEG.config.CNN_LEARNING_RATE
        else:
            learning_rate = float(self.ui.textEdit_learning_rate)

        if self.ui.textEdit_electrodes.toPlainText().strip() == "":
            electrodes = EEG.config.EEG_NUM_OF_ELECTRODES
        else:
            electrodes = int(self.ui.textEdit_electrodes)

        if self.ui.textEdit_frame_size.toPlainText().strip() == "":
            frame_size = EEG.config.EEG_SIGNAL_FRAME_SIZE
        else:
            frame_size = int(self.ui.textEdit_frame_size)

        if self.ui.textEdit_frequency.toPlainText().strip() == "":
            frequency = EEG.config.FS
        else:
            frequency = int(self.ui.textEdit_frequency)

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

        #self.real_time_metrics = RealTimeMetrics(epochs, self.ui.plotLabel_CNN)
        #self.real_time_metrics.start()

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
        train_cnn_eeg_readraw(True, self.pathTrain, PREDICT_PATH, MODEL_PATH)

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

class RealTimeMetrics(QThread):
    """Thread for visualizing accuracy and loss in real time during model training."""

    def __init__(self, total_epochs, plot_label):
        super().__init__()
        self.total_epochs = total_epochs
        self.plot_label = plot_label
        self.mutex = QMutex()

    def run(self):
        while global_epoch_count <= self.total_epochs:
            self.plot_metrics()
            time.sleep(10)

    def plot_metrics(self):
        try:
            with QMutexLocker(self.mutex):
                fig = Figure()
                fig.tight_layout()
                canvas = FigureCanvas(fig)

                print()
                print("\nTest 2: ")
                print("global_accuracy: ", global_accuracy)
                print("global_val_accuracy: ", global_val_accuracy)
                print("global_loss: ", global_loss)
                print("global_val_loss: ", global_val_loss)
                print("global_epoch_count: ", global_epoch_count,"\n")
                print()

                # Plot for accuracy
                ax1 = fig.add_subplot(211)
                ax1.plot(global_epoch_count, global_accuracy, 'r-', label='Training Accuracy')
                ax1.plot(global_epoch_count, global_val_accuracy, 'b-', label='Validation Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Accuracy')
                ax1.legend()
                ax1.grid(True)
                ax1.set_ylim(0, 1.0)
                ax1.set_xlim(1, self.total_epochs)

                # Plot for loss
                ax2 = fig.add_subplot(212)
                ax2.plot(global_epoch_count, global_loss, 'r-', label='Training Loss')
                ax2.plot(global_epoch_count, global_val_loss, 'b-', label='Validation Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Loss')
                ax2.legend()
                ax2.grid(True)
                ax2.set_xlim(1, self.total_epochs)

                fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing

                buf = io.BytesIO()
                canvas.print_png(buf)
                qpm = QPixmap()
                qpm.loadFromData(buf.getvalue(), 'PNG')
                self.plot_label.setPixmap(qpm)

                buf.close()
        except Exception as e:
            print(f"Wystąpił błąd podczas tworzenia wykresu: {e}")


