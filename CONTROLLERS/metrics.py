import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tensorflow.keras.callbacks import Callback
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
import io


class RealTimeMetrics(Callback):
    """Callback for visualizing accuracy and loss in real time during model training.

    Args:
        total_epochs (int): Total number of training epochs.
        plot_label (QLabel): QLabel widget where the plots will be displayed.
    """
    def __init__(self, total_epochs, plot_label):
        super().__init__()
        self.epoch_count = 0
        self.total_epochs = total_epochs
        self.plot_label = plot_label
        self.x_data = []
        self.y_data = []
        self.val_y_data = []
        self.loss_data = []
        self.val_loss_data = []
        self.METRICS_DISP_INTERVAL = 5

    def on_epoch_end(self, epoch, logs=None):
        """Method called at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of logs containing metrics.
        """
        logs = logs or {}
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        self.x_data.append(self.epoch_count)
        self.y_data.append(accuracy)
        self.val_y_data.append(val_accuracy)
        self.loss_data.append(loss)
        self.val_loss_data.append(val_loss)

        if self.epoch_count % self.METRICS_DISP_INTERVAL == 0 or self.epoch_count == self.total_epochs - 1:
            self.plot_metrics()

        self.epoch_count += 1

    def plot_metrics(self):
        try:
            fig = Figure(figsize=(12, 6))
            fig.tight_layout()
            canvas = FigureCanvas(fig)

            ax1 = fig.add_subplot(121)
            ax1.plot(self.x_data, self.y_data, 'r-', label='Training Accuracy')
            ax1.plot(self.x_data, self.val_y_data, 'b-', label='Validation Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy')
            ax1.legend()
            ax1.grid(True)
            ax1.set_ylim(0, 1.0)
            ax1.set_xlim(0, self.total_epochs)

            ax2 = fig.add_subplot(122)
            ax2.plot(self.x_data, self.loss_data, 'r-', label='Training Loss')
            ax2.plot(self.x_data, self.val_loss_data, 'b-', label='Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlim(0, self.total_epochs)

            buf = io.BytesIO()
            canvas.print_png(buf)
            qpm = QPixmap()
            qpm.loadFromData(buf.getvalue(), 'PNG')
            self.plot_label.setPixmap(qpm)

            buf.close()
        except Exception as e:
            print(f"Wystąpił błąd podczas tworzenia wykresu: {e}")

    def on_train_end(self, logs=None):
        plt.ioff()
        self.plot_metrics()
