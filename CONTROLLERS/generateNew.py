import io
import os
import sys
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QThread, QModelIndex, QSize
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPixmap, QMovie
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from CONTROLLERS.DBConnector import DBConnector

class ModelWorker(QObject):
    """
    Worker class to load the model in a separate thread.

    Attributes:
        finished (pyqtSignal): Signal emitted when the model is loaded.
        error (pyqtSignal): Signal emitted when an error occurs.
        model_loaded (pyqtSignal): Signal emitted with the loaded model.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    model_loaded = pyqtSignal(object)

    def __init__(self, db, model_name):
        """
        Initializes the ModelWorker class with the database connector and model name.

        Args:
            db: The database connector instance.
            model_name: The name of the model to load.
        """
        super().__init__()
        self.db = db
        self.model_name = model_name

    def run(self):
        """
        Loads the model from the database. Emits model_loaded signal if successful,
        and error signal if an exception occurs.

        This method is intended to be run in a separate thread.
        """
        try:
            model = self.db.select_model(self.model_name)
            self.model_loaded.emit(model)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


def get_base_path():
    """
    Returns:
        str: The base path of the application.
    """
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(current_directory)

class GenerateNew:
    def __init__(self, ui):
        """
        Initialize the GenerateNew class.

        Args:
            ui: User interface instance.
        """
        self.ui = ui
        self.db = DBConnector()
        self.chosen_model_data = None
        self.choose_model()

        self.ui.imgNumberBox.setRange(1, 20)
        self.generated = []
        self.currIdxMRI = 0
        self.generator = None
        self.thread = None
        self.worker = None
        self.fig = None
        self.gif_path = os.path.join(get_base_path() ,'UI', 'loading.gif')

    def __del__(self):
        """
        Delete the generator and generated images.
        """
        if self.generated is not None:
            del self.generated
        if self.generator is not None:
            del self.generator

    def on_exit(self):
        """
        Quit the application
        """
        QApplication.quit()

    def choose_model(self):
        """
        Choose a model for image generation.
        """
        def choose_adhd_model(index: QModelIndex):
            item = adhd_model.itemFromIndex(index)
            self.chosen_model_data = item.data()
            self.ui.adhdGenName.setText(self.chosen_model_data[0])

            self.ui.controlGenList.clearSelection()
            self.ui.controlGenName.setText("---------------------------")

        def choose_control_model(index: QModelIndex):
            item = control_model.itemFromIndex(index)
            self.chosen_model_data = item.data()
            self.ui.controlGenName.setText(self.chosen_model_data[0])

            self.ui.adhdGenList.clearSelection()
            self.ui.adhdGenName.setText("---------------------------")

        try:
            self.db.establish_connection()
        except ConnectionError as e:
            self.show_alert(str(e))
            return

        adhd_model = QStandardItemModel()
        adhd_list = self.db.select_model_info("type='gan_adhd'")

        for item in adhd_list:
            item = list(item)
            item.append('adhd')
            adhd_item = QStandardItem(item[0])
            adhd_item.setEditable(False)
            adhd_item.setData(item)
            adhd_model.appendRow(adhd_item)
        self.ui.adhdGenList.setModel(adhd_model)
        self.ui.adhdGenList.doubleClicked.connect(choose_adhd_model)

        control_model = QStandardItemModel()
        control_list = self.db.select_model_info("type='gan_control'")

        for item in control_list:
            item = list(item)
            item.append('healthy')
            control_item = QStandardItem(item[0])
            control_item.setEditable(False)
            control_item.setData(item)
            control_model.appendRow(control_item)
        self.ui.controlGenList.setModel(control_model)
        self.ui.controlGenList.doubleClicked.connect(choose_control_model)

    def show_info(self, data):
        """
        Display information about the selected generator model.

        Args:
            data (str): Type of data, 'adhd' or 'control'.
        """
        if data == "adhd" and self.ui.adhdGenName.text() == "---------------------------":
            return
        elif data == "control" and self.ui.controlGenName.text() == "---------------------------":
            return

        alert = QMessageBox()
        alert.setWindowTitle("Info of generator model")
        alert.setIcon(QMessageBox.Information)
        alert.setStandardButtons(QMessageBox.Ok)

        plane = self.chosen_model_data[4]
        msg = f"""
        Generator loss: {self.chosen_model_data[0]}\n
        Image size: {self.chosen_model_data[1]}\n
        Plane: {'Axial' if plane == 'A' else 'Sagittal' if plane == 'S' else 'Coronal'}\n
        Description: {self.chosen_model_data[5]}
        """
        alert.setText(msg.strip())
        alert.exec_()

    def show_alert(self, msg):
        """
        Display a warning message.

        Args:
            msg (str): The content of the warning message.
        """
        alert = QMessageBox()
        alert.setWindowTitle("Warning")
        alert.setText(msg)
        alert.setIcon(QMessageBox.Warning)
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec_()

    def handle_error(self, error_msg):
        """
        Handle errors during the process.

        Args:
            error_msg (str): The error message to display.
        """
        self.show_alert(f"Error: {error_msg}")

    def generate(self):
        """
        Generate MRI images using the selected model.
        """
        self.generated = []
        self.currIdxMRI = 0

        if self.chosen_model_data is None:
            self.show_alert("Please select model!")
            return

        self.thread = QThread()
        self.worker = ModelWorker(self.db, self.chosen_model_data[0])
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.worker.error.connect(self.handle_error)

        self.thread.start()
        self.show_loading_animation()
        self.ui.genBtn.setEnabled(False)
        self.ui.backBtn.setEnabled(False)

    def on_model_loaded(self, model):
        """
        Callback when the model is loaded. Generates images using the loaded model.

        Args:
            model: The loaded model.
        """
        self.generator = model

        img_amount = self.ui.imgNumberBox.value()
        for _ in range(img_amount):
            noise = np.random.normal(0, 1, [1, 100])
            generated_image = self.generator.predict(noise)
            generated_image = generated_image * 0.5 + 0.5
            self.generated.append(generated_image[0])

        self.show_plot_mri(self.generated[0])
        self.ui.genBtn.setEnabled(True)
        self.ui.backBtn.setEnabled(True)

    def show_prev_plot_mri(self):
        """
        Display the previous generated MRI image.
        """
        if len(self.generated) == 0: return

        self.currIdxMRI -= 1

        if self.currIdxMRI < 0:
            self.currIdxMRI = 0

        self.show_plot_mri(self.generated[self.currIdxMRI])

    def show_next_plot_mri(self):
        """
        Display the next generated MRI image.
        """
        if len(self.generated) == 0: return

        self.currIdxMRI += 1

        if self.currIdxMRI > len(self.generated) - 1:
            self.currIdxMRI = len(self.generated) - 1

        self.show_plot_mri(self.generated[self.currIdxMRI])

    def show_plot_mri(self, img):
        """
        Display the generated MRI image.

        Args:
            img (np.ndarray): Generated MRI image.
        """
        self.fig = Figure()
        self.fig.tight_layout()
        canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.ax.imshow(img, cmap="gray")
        self.ax.set_title(f'MRI generated {self.currIdxMRI + 1}')

        buf = io.BytesIO()
        canvas.print_png(buf)
        qpm = QPixmap()
        qpm.loadFromData(buf.getvalue(), 'PNG')
        self.ui.plotLabelMRI.setPixmap(qpm)

    def show_loading_animation(self):
        self.movie = QMovie(self.gif_path)
        self.ui.plotLabelMRI.setMovie(self.movie)
        self.movie.setScaledSize(QSize(50, 50))
        self.movie.start()

    def save_image(self):
        if self.fig is None: return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self.ui.centralwidget, "Save Figure", "", "PNG Files (*.png);;All Files (*)", options=options)

        if file_path:
            extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(file_path, bbox_inches=extent)
