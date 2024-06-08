from tensorflow.keras.callbacks import Callback
from PyQt5.QtCore import QMutex, QMutexLocker, QThread
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QProgressBar

import io
import time


# Global variables for metrics
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



    def run(self):
        self.clear_metrics()
        control_counter = 0
        while control_counter < self.total_epochs:
            control_counter = len(global_accuracy)
            self.plot_metrics()
            time.sleep(self.interval)
            self.progressBar.setValue(control_counter)


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





