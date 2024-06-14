from tensorflow.keras.callbacks import Callback
from PyQt5.QtCore import QMutex, QMutexLocker, QThread
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5.QtWidgets import QPushButton

from PyQt5.QtGui import QPixmap

import numpy as np
from matplotlib import pyplot as plt


import io
import time


global_train_d_loss = []
global_train_g_loss = []
global_val_d_loss = []
global_val_g_loss = []

global_accuracy = []
global_loss = []
global_val_loss = []

generated_image = []

def plot_mri(image, title):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

class RealTimeMetrics(QThread):
    """Thread for visualizing accuracy and loss in real time during model training."""
    def __init__(self, total_epochs, progressBar, plot_label, interval=1):
        super().__init__()
        self.total_epochs = total_epochs
        self.plot_label = plot_label
        self.plot_label.setScaledContents(True)
        self.plot_label.setMinimumSize(640, 640)
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
                self.progressBar.setValue(0)
                break
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
        global global_accuracy, global_val_accuracy, global_loss, global_val_loss, generated_image
        global_accuracy = []
        global_val_accuracy = []
        global_loss = []
        global_val_loss = []
        generated_image = []

class RealTimeMetrics_GEN(QThread):
    """Thread for visualizing GAN metrics in real time during model training."""
    def __init__(self, total_epochs, print_interval, disp_interval, plot_label, image_label, interval=1):
        super().__init__()
        self.print_interval = print_interval
        self.disp_interval = disp_interval
        self.total_epochs = total_epochs
        self.plot_label = plot_label
        self.plot_label.setScaledContents(True)
        self.plot_label.setMinimumSize(640, 640)
        self.image_label = image_label
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(640, 640)
        self.mutex = QMutex()
        self.interval = interval
        self.running = True

    def run(self):
        self.clear_metrics()
        control_counter = 0
        while control_counter < self.total_epochs:
            control_counter = len(global_train_d_loss)*self.print_interval
            disp_counter = self.disp_interval*((len(global_train_d_loss)*self.print_interval)//self.disp_interval)
            self.plot_metrics()
            self.generate_and_display_image(disp_counter)
            time.sleep(self.interval)

    def generate_and_display_image(self, epoch):
        if not generated_image == []:
            try:
                buf = plot_mri(generated_image[0], f'Epoch: {epoch}')
                qpm = QPixmap()
                qpm.loadFromData(buf.getvalue(), 'PNG')
                self.image_label.setPixmap(qpm)
                buf.close()
            except Exception as e:
                print(f"Failed to generate image in epoch {epoch}: {e}")

    def plot_metrics(self):
        try:
            with QMutexLocker(self.mutex):
                fig = Figure()
                canvas = FigureCanvas(fig)

                # Plot for generator loss
                ax1 = fig.add_subplot(211)
                ax1.plot(range(1, len(global_train_g_loss)*self.print_interval + 1, self.print_interval), global_train_g_loss, 'r-', label='Training Generator Loss')
                ax1.plot(range(1, len(global_val_g_loss)*self.print_interval + 1, self.print_interval), global_val_g_loss, 'b-', label='Validation Generator Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Generator Loss')
                ax1.legend()
                ax1.grid(True)
                ax1.set_xlim(1, self.total_epochs)

                # Plot for discriminator loss
                ax2 = fig.add_subplot(212)
                ax2.plot(range(1, len(global_train_d_loss)*self.print_interval + 1, self.print_interval), global_train_d_loss, 'r-', label='Training Discriminator Loss')
                ax2.plot(range(1, len(global_val_d_loss)*self.print_interval + 1, self.print_interval), global_val_d_loss, 'b-', label='Validation Discriminator Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Discriminator Loss')
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
        global global_train_d_loss, global_train_g_loss
        global global_val_d_loss, global_val_g_loss
        global_train_d_loss = []
        global_train_g_loss = []
        global_val_d_loss = []
        global_val_g_loss = []

class WorkerMetrics(Callback):
    """Callback for updating global metrics during model training."""
    def __init__(self, total_epochs):
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
        self.val_d_loss = []
        self.val_g_loss = []

    def update_train_metrics(self, d_loss, g_loss):
        self.train_d_loss.append(d_loss)
        self.train_g_loss.append(g_loss)
        self.update_global_metrics()

    def update_val_metrics(self, d_loss, g_loss):
        self.val_d_loss.append(d_loss)
        self.val_g_loss.append(g_loss)
        self.update_global_metrics()

    def get_metrics(self):
        print("\ntrain_d_loss", global_train_d_loss,)
        print("\ntrain_g_loss", global_train_g_loss,)
        print("\nval_d_loss", global_val_d_loss,)
        print("\nval_g_loss", global_val_g_loss,)

    def generate_image(self, generator, epoch):
        try:
            global generated_image
            noise = np.random.normal(0, 1, [1, 100])
            generated_image_predict = generator.predict(noise)
            generated_image = generated_image_predict * 0.5 + 0.5  # Scale the image to [0, 1]

        except Exception as e:
            print(f"Failed to generate image in epoch {epoch}: {e}")

    def update_global_metrics(self):
        global global_train_d_loss, global_train_g_loss
        global global_val_d_loss, global_val_g_loss
        global_train_d_loss = self.train_d_loss
        global_train_g_loss = self.train_g_loss
        global_val_d_loss = self.val_d_loss
        global_val_g_loss = self.val_g_loss



