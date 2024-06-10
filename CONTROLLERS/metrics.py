from tensorflow.keras.callbacks import Callback
from PyQt5.QtCore import QMutex, QMutexLocker, QThread
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QProgressBar

import io
import time

global_train_d_loss = []
global_train_g_loss = []
global_train_d_accuracy = []
global_val_d_loss = []
global_val_g_loss = []
global_val_d_accuracy = []

global_accuracy = []
global_val_accuracy = []
global_loss = []
global_val_loss = []

class RealTimeMetrics(QThread):
    """Thread for visualizing accuracy and loss in real time during model training."""
    def __init__(self, total_epochs, progressBar, plot_label, interval=1):
        super().__init__()

        self.total_epochs = total_epochs
        self.plot_label = plot_label
        self.mutex = QMutex()
        self.interval = interval
        self.progressBar = progressBar
        self.progressBar.setRange(0, total_epochs)

    def run(self):
        self.clear_metrics()
        control_counter = 0
        while control_counter < self.total_epochs:
            control_counter = len(global_accuracy)
            self.plot_metrics()
            time.sleep(self.interval)
            self.progressBar.setValue(control_counter)

    def stop(self):
        self.running = False

    def plot_metrics(self):
        try:
            with QMutexLocker(self.mutex):
                fig = Figure()
                canvas = FigureCanvas(fig)

                # Plot for accuracy
                ax1 = fig.add_subplot(211)
                ax1.plot(range(1, len(global_accuracy) + 1), global_accuracy, 'r-', label='Training Accuracy')
                ax1.plot(range(1, len(global_val_accuracy) + 1), global_val_accuracy, 'b-', label='Validation Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Accuracy')
                ax1.legend()
                ax1.grid(True)
                ax1.set_ylim(0, 1.0)
                ax1.set_xlim(1, self.total_epochs)

                # Plot for loss
                ax2 = fig.add_subplot(212)
                ax2.plot(range(1, len(global_loss) + 1), global_loss, 'r-', label='Training Loss')
                ax2.plot(range(1, len(global_val_loss) + 1), global_val_loss, 'b-', label='Validation Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Loss')
                ax2.legend()
                ax2.grid(True)
                ax2.set_xlim(1, self.total_epochs)

                fig.tight_layout()
                fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing

                buf = io.BytesIO()
                canvas.print_png(buf)
                qpm = QPixmap()
                qpm.loadFromData(buf.getvalue(), 'PNG')
                self.plot_label.setPixmap(qpm)

                buf.close()
        except Exception as e:
            print(f"An error occurred while creating the plot: {e}")

    def clear_metrics(self):
        global global_accuracy, global_val_accuracy, global_loss, global_val_loss
        global_accuracy = []
        global_val_accuracy = []
        global_loss = []
        global_val_loss = []

class RealTimeMetrics_GEN(QThread):
    """Thread for visualizing GAN metrics in real time during model training."""
    def __init__(self, total_epochs, progressBar, plot_label, interval=1):
        super().__init__()

        self.total_epochs = total_epochs
        self.plot_label = plot_label
        self.mutex = QMutex()
        self.interval = interval
        self.progressBar = progressBar
        self.progressBar.setRange(0, total_epochs)
        self.running = True

    def run(self):
        self.clear_metrics()
        control_counter = 0
        while control_counter < self.total_epochs:
            if not self.running:
                break
            control_counter = len(global_train_d_loss)
            self.plot_metrics()
            time.sleep(self.interval)
            self.progressBar.setValue(control_counter)

    def stop(self):
        self.running = False

    def plot_metrics(self):
        try:
            with QMutexLocker(self.mutex):
                fig = Figure()
                canvas = FigureCanvas(fig)

                # Plot for discriminator accuracy
                ax1 = fig.add_subplot(311)
                ax1.plot(range(1, len(global_train_d_accuracy) + 1), global_train_d_accuracy, 'r-', label='Training Discriminator Accuracy')
                ax1.plot(range(1, len(global_val_d_accuracy) + 1), global_val_d_accuracy, 'b-', label='Validation Discriminator Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.set_title('Discriminator Accuracy')
                ax1.legend()
                ax1.grid(True)
                ax1.set_ylim(0, 1.0)
                ax1.set_xlim(1, self.total_epochs)

                # Plot for generator loss
                ax2 = fig.add_subplot(312)
                ax2.plot(range(1, len(global_train_g_loss) + 1), global_train_g_loss, 'r-', label='Training Generator Loss')
                ax2.plot(range(1, len(global_val_g_loss) + 1), global_val_g_loss, 'b-', label='Validation Generator Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Generator Loss')
                ax2.legend()
                ax2.grid(True)
                ax2.set_xlim(1, self.total_epochs)

                # Plot for discriminator loss
                ax3 = fig.add_subplot(313)
                ax3.plot(range(1, len(global_train_d_loss) + 1), global_train_d_loss, 'r-', label='Training Discriminator Loss')
                ax3.plot(range(1, len(global_val_d_loss) + 1), global_val_d_loss, 'b-', label='Validation Discriminator Loss')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.set_title('Discriminator Loss')
                ax3.legend()
                ax3.grid(True)
                ax3.set_xlim(1, self.total_epochs)

                fig.tight_layout()
                fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing

                buf = io.BytesIO()
                canvas.print_png(buf)
                qpm = QPixmap()
                qpm.loadFromData(buf.getvalue(), 'PNG')
                self.plot_label.setPixmap(qpm)

                buf.close()
        except Exception as e:
            print(f"An error occurred while creating the plot: {e}")

    def clear_metrics(self):
        global global_train_d_loss, global_train_g_loss, global_train_d_accuracy
        global global_val_d_loss, global_val_g_loss, global_val_d_accuracy
        global_train_d_loss = []
        global_train_g_loss = []
        global_train_d_accuracy = []
        global_val_d_loss = []
        global_val_g_loss = []
        global_val_d_accuracy = []

class WorkerMetrics(Callback):
    """Callback for updating global metrics during model training."""
    def __init__(self,total_epochs):
        super().__init__()
        self.epoch_counter = 0
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        global global_accuracy, global_val_accuracy, global_loss, global_val_loss
        logs = logs or {}
        global_accuracy.append(logs.get('accuracy'))
        global_val_accuracy.append(logs.get('val_accuracy'))
        global_loss.append(logs.get('loss'))
        global_val_loss.append(logs.get('val_loss'))
        self.epoch_counter += 1


class WorkerMetrics_GAN:
    def __init__(self):
        self.train_d_loss = []
        self.train_g_loss = []
        self.train_d_accuracy = []
        self.val_d_loss = []
        self.val_g_loss = []
        self.val_d_accuracy = []

    def update_train_metrics(self, d_loss, g_loss, d_accuracy):
        self.train_d_loss.append(d_loss)
        self.train_g_loss.append(g_loss)
        self.train_d_accuracy.append(d_accuracy)
        self.update_global_metrics()
        self.get_metrics()

    def update_val_metrics(self, d_loss, g_loss, d_accuracy):
        self.val_d_loss.append(d_loss)
        self.val_g_loss.append(g_loss)
        self.val_d_accuracy.append(d_accuracy)
        self.update_global_metrics()
        self.get_metrics()

    def get_metrics(self):
        return {
            "train_d_loss": self.train_d_loss,
            "train_g_loss": self.train_g_loss,
            "train_d_accuracy": self.train_d_accuracy,
            "val_d_loss": self.val_d_loss,
            "val_g_loss": self.val_g_loss,
            "val_d_accuracy": self.val_d_accuracy,
        }

    def update_global_metrics(self):
        global global_train_d_loss, global_train_g_loss, global_train_d_accuracy
        global global_val_d_loss, global_val_g_loss, global_val_d_accuracy
        global_train_d_loss = self.train_d_loss
        global_train_g_loss = self.train_g_loss
        global_train_d_accuracy = self.train_d_accuracy
        global_val_d_loss = self.val_d_loss
        global_val_g_loss = self.val_g_loss
        global_val_d_accuracy = self.val_d_accuracy
