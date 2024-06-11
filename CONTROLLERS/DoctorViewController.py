import os
import io
import random
import ast
import numpy as np
import nibabel as nib
import pyedflib
from PyQt5 import uic
from PyQt5.QtCore import QModelIndex, QThread, QObject, pyqtSignal, QSize, Qt
from PyQt5.QtWidgets import (
    QFileDialog, QDialog, QVBoxLayout, QRadioButton, QLineEdit, QLabel,
    QPushButton, QMessageBox
)
from PyQt5.QtGui import QPixmap, QStandardItem, QStandardItemModel, QIntValidator, QMovie
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.io import loadmat
from scipy.ndimage import rotate
from pandas import read_csv

from keras.models import load_model

import EEG.config
from CONTROLLERS.DBConnector import DBConnector
from EEG.data_preprocessing import (
    filter_eeg_data, clip_eeg_data, normalize_eeg_data
)
from EEG.file_io import split_into_frames
from EEG.PREDICT.predict import check_result
from MRI.file_io import read_pickle
from MRI.image_preprocessing import trim_one, normalize
import sys
def get_base_path():
    """
    Returns:
        str: The base path of the application.
    """
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

current_dir = os.path.dirname(__file__)
parent_directory = os.path.dirname(get_base_path())
UI_PATH = os.path.join(get_base_path(), 'UI')
FILE_TYPES = ["mat", "csv", 'edf', 'nii.gz', 'nii']
GIF_PATH = os.path.join(UI_PATH, 'loading.gif')
electrode_positions = [
    "Fz", "Cz", "Pz", "C3", "T3", "C4", "T4", "Fp1", "Fp2", "F3", "F4",
    "F7", "F8", "P3", "P4", "T5", "T6", "O1", "O2"]

class DoctorViewController:
    """
    Initialization and Setup
    """

    def __init__(self, main_window, ui_path, main_path):
        self.MAIN_PATH = main_path
        """
        Initializes the doctor view controller, loads the UI, and adds events.

        :param main_window: The main window of the application.
        """
        self.main_window = main_window
        self.main_window.setWindowTitle("DOCTOR VIEW")
        self.ui = uic.loadUi(
            os.path.join(ui_path, 'doctorView.ui'),
            main_window
        )
        self.add_events()

        self.db_conn = None
        self.file_paths = None
        self.model_eeg = None
        self.model_mri = None
        self.chosen_model_info_eeg = None
        self.chosen_model_info_mri = None
        self.loaded_eeg_files = 0
        self.loaded_mri_files = 0
        self.curr_idx_eeg = 0
        self.curr_idx_mri = 0
        self.curr_idx_channel = 0
        self.curr_idx_plane = 0
        self.predictions = None
        self.all_data = {"EEG": {
            "data" : [],
            "names" : []
        }, "MRI": {
            "data" : [],
            "names" : []
        }}

    def add_events(self):
        """
        Adds events to the buttons and UI elements.
        """
        self.ui.loadDataBtn.clicked.connect(self.get_file_paths)
        self.ui.btnNextPlot.clicked.connect(self.show_next_plot_eeg)
        self.ui.btnPrevPlot.clicked.connect(self.show_prev_plot_eeg)
        self.ui.btnNextChannel.clicked.connect(self.show_next_channel)
        self.ui.btnPrevChannel.clicked.connect(self.show_prev_channel)
        self.ui.btnNextPlot_2.clicked.connect(self.show_next_plot_mri)
        self.ui.btnPrevPlot_2.clicked.connect(self.show_prev_plot_mri)
        self.ui.btnNextPlane.clicked.connect(self.show_next_plane)
        self.ui.btnPrevPlane.clicked.connect(self.show_prev_plane)
        self.ui.modelInfoEEG.clicked.connect(lambda: self.show_model_info("EEG"))
        self.ui.modelInfoMRI.clicked.connect(lambda: self.show_model_info("MRI"))
        self.ui.predictBtn.clicked.connect(self.predict)
        self.ui.showReal.clicked.connect(lambda: self.show_dialog('REAL'))
        self.ui.showGenerated.clicked.connect(lambda: self.show_dialog('GENERATED'))

    def get_file_paths(self):
        """
        Opens a file selection dialog and saves the paths of the selected files.
        """
        options = QFileDialog.Options()
        self.file_paths, _ = QFileDialog.getOpenFileNames(
            self.main_window, "Choose files", "", "", options=options
        )

        if len(self.file_paths) == 0:
            self.file_paths = None
            return

        self.loaded_eeg_files = 0
        self.loaded_mri_files = 0

        for path in self.file_paths:
            if path.endswith('.mat') or path.endswith('.edf') or path.endswith('.csv'):
                self.loaded_eeg_files += 1
            if path.endswith('.nii') or path.endswith('.nii.gz'):
                self.loaded_mri_files += 1

        if self.loaded_eeg_files > 0:
            self.ui.imgViewer.setCurrentIndex(0)
        if self.loaded_mri_files > 0:
            self.ui.imgViewer.setCurrentIndex(1)

        self.ui.dataName.setText(f"{self.loaded_eeg_files} EEG and {self.loaded_mri_files} MRI files chosen")
        self.get_model_names()

    def get_model_names(self):
        """
        Fetches available models from the database and displays them in the UI.
        """
        self.chosen_model_info_eeg = None
        self.chosen_model_info_mri = None
        self.ui.chosenModelEEG.setText("----------")
        self.ui.chosenModelMRI.setText("----------")

        if self.db_conn is None:
            self.db_conn = DBConnector()
            try:
                self.db_conn.establish_connection()
            except ConnectionError:
                self.show_alert("Cannot establish database connection, remember to enable ZUT VPN.")
                return

        model_eeg = self.ui.modelListViewEEG.model()
        if model_eeg:
            model_eeg.clear()
        else:
            model_eeg = QStandardItemModel()

        if self.loaded_eeg_files > 0:
            def choose_model_eeg(index: QModelIndex):
                item = model_eeg.itemFromIndex(index)
                self.chosen_model_info_eeg = item.data()
                self.ui.chosenModelEEG.setText(self.chosen_model_info_eeg[0])

            models_list = self.db_conn.select_model_info("type='cnn_eeg'")

            for model_info in models_list:
                item = QStandardItem(model_info[0])
                item.setEditable(False)
                item.setData(model_info)
                model_eeg.appendRow(item)

            self.ui.modelListViewEEG.setModel(model_eeg)
            self.ui.modelListViewEEG.doubleClicked.connect(choose_model_eeg)

        model_mri = self.ui.modelListViewMRI.model()
        if model_mri:
            model_mri.clear()
        else:
            model_mri = QStandardItemModel()

        if self.loaded_mri_files:
            def choose_model_mri(index: QModelIndex):
                item = model_mri.itemFromIndex(index)
                self.chosen_model_info_mri = item.data()
                self.ui.chosenModelMRI.setText(self.chosen_model_info_mri[0])

            models_list = self.db_conn.select_model_info("type='cnn_mri'")

            for model_info in models_list:
                item = QStandardItem(model_info[0])
                item.setEditable(False)
                item.setData(model_info)
                model_mri.appendRow(item)

            self.ui.modelListViewMRI.setModel(model_mri)
            self.ui.modelListViewMRI.doubleClicked.connect(choose_model_mri)

    def load_models(self):
        """
        Loads the selected models from the database.
        """
        self.model_eeg = None
        self.model_mri = None

        if self.chosen_model_info_eeg is not None:
            self.model_eeg = self.db_conn.select_model(self.chosen_model_info_eeg[0])
        if self.chosen_model_info_mri is not None:
            self.model_mri = self.db_conn.select_model(self.chosen_model_info_mri[0])

    """
    Dialogs and Alerts
    """

    def show_dialog(self, data_type):
        """
        Displays a dialog to choose options for generated or real MRI data.

        :param data_type: The type of data ('REAL' or 'GENERATED').
        """
        dialog = QDialog()
        dialog.setWindowTitle('Choose option')
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        layout = QVBoxLayout()
        if data_type == 'GENERATED':
            label_desc = QLabel('MRI pictures generated on default, build in model.')
            layout.addWidget(label_desc)

        radio_adhd = QRadioButton('ADHD')
        radio_control = QRadioButton('CONTROL')
        radio_adhd.setChecked(True)

        layout.addWidget(radio_adhd)
        layout.addWidget(radio_control)

        label = QLabel('IMG amount (max 20):')
        layout.addWidget(label)

        input_number = QLineEdit()
        validator = QIntValidator(0, 20, input_number)
        input_number.setValidator(validator)
        input_number.setText("3")
        layout.addWidget(input_number)

        submit_button = QPushButton('Submit')
        submit_button.clicked.connect(
            lambda: self.prepare_and_plot_data(
                data_type, radio_adhd, input_number, dialog))
        layout.addWidget(submit_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_alert(self, msg):
        """
        Displays a warning message.

        :param msg: The warning message.
        """
        alert = QMessageBox()
        alert.setWindowTitle("Warning")
        alert.setText(msg)
        alert.setIcon(QMessageBox.Warning)
        alert.setStandardButtons(QMessageBox.Ok)
        alert.exec_()

    def show_model_info(self, model_type):
        """
        Displays information about the selected model.

        :param model_type: The type of model ('EEG' or 'MRI').
        """
        alert = QMessageBox()
        alert.setWindowTitle("Info")
        alert.setIcon(QMessageBox.Information)
        alert.setStandardButtons(QMessageBox.Ok)

        if model_type == "EEG":
            if self.chosen_model_info_eeg is None:
                return
            msg = f"""
                Model accuracy: {self.chosen_model_info_eeg[0]}\n
                Input shape: {self.chosen_model_info_eeg[1]}\n
                Frequency: {self.chosen_model_info_eeg[2]}\n
                Channels: {self.chosen_model_info_eeg[3]}\n
                Description: {self.chosen_model_info_eeg[5]}
                """
        elif model_type == "MRI":
            if self.chosen_model_info_mri is None:
                return
            plane = self.chosen_model_info_mri[4]
            msg = f"""
                Model accuracy: {self.chosen_model_info_mri[0]}\n
                Input shape: {self.chosen_model_info_mri[1]}\n
                Plane: {'Axial' if plane == 'A' else 'Sagittal' if plane == 'S' else 'Coronal'}\n
                Description: {self.chosen_model_info_mri[5]}
                """
        alert.setText(msg)
        alert.exec_()

    """
    Prediction and Processing
    """

    def predict(self):
        """
        Starts the prediction process based on the loaded data and models.
        """
        self.ui.btnNextPlane.setEnabled(True)
        self.ui.btnPrevPlane.setEnabled(True)

        if self.file_paths is None or (self.chosen_model_info_eeg is None and self.chosen_model_info_mri is None):
            self.show_alert("No files or models chosen")
            return

        self.curr_idx_eeg = 0
        self.curr_idx_mri = 0
        self.curr_idx_channel = 0
        self.curr_idx_plane = 0

        self.thread = QThread()
        self.worker = Worker(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.on_error)

        self.thread.start()
        self.show_loading_animation()
        self.change_btn_state(False)

    def process_files(self):
        """
        Processes the loaded files, making predictions based on the selected models.
        """
        self.predictions = []
        self.all_data = {"EEG": {
            "data": [],
            "names": []
        }, "MRI": {
            "data": [],
            "names": []
        }}

        for path in self.file_paths:
            data = np.array([])
            data_type = ""

            if path.endswith('.edf'):
                f = pyedflib.EdfReader(path)
                n_channels = f.signals_in_file
                n_samples = f.getNSamples()[0]
                data = np.zeros((n_channels, n_samples))
                for i in range(n_channels):
                    data[i, :] = f.readSignal(i)
                f.close()
                data_type = "EEG"
                model = self.model_eeg
                model_info = self.chosen_model_info_eeg

            elif path.endswith('.mat'):
                file = loadmat(path)
                data_key = list(file.keys())[-1]
                data = file[data_key].T
                data_type = "EEG"
                model = self.model_eeg
                model_info = self.chosen_model_info_eeg

            elif path.endswith('.csv'):
                data = read_csv(path).values.T
                data_type = "EEG"
                model = self.model_eeg
                model_info = self.chosen_model_info_eeg

            elif path.endswith('.nii.gz') or path.endswith('.nii'):
                file = nib.load(path)
                file_data = file.get_fdata()
                (x, y, z, t) = file_data.shape

                frontal_plane = file_data[:, int(y / 2), :, int(t / 2)]
                sagittal_plane = file_data[int(x / 2), :, :, int(t / 2)]
                horizontal_plane = file_data[:, :, int(z / 2), int(t / 2)]

                data = horizontal_plane
                data_type = "MRI"
                self.all_data[data_type]["data"].append([horizontal_plane,
                                                 rotate(sagittal_plane, 90, reshape=True),
                                                 rotate(frontal_plane, 90, reshape=True)])
                model = self.model_mri
                model_info = self.chosen_model_info_mri

            result = self.process_data(data, model, model_info, data_type=data_type)

            if data_type == "EEG":
                self.all_data[data_type]["data"].append(data)

            self.all_data[data_type]["names"].append(path)

            if result is not None:
                self.predictions.append(result)

    def process_data(self, data, model, model_info, data_type="EEG"):
        """
        Processes EEG or MRI data and makes predictions using the model.

        :param data: The data to process.
        :param model: The model to use for predictions.
        :param model_info: Information about the model.
        :param data_type: The type of data ('EEG' or 'MRI').
        :return: The prediction result.
        """
        from MRI.config import CNN_INPUT_SHAPE_MRI

        result = []

        if data_type == "EEG":
            input_shape = ast.literal_eval(model_info[1])
            channels = input_shape[0]
            EEG.config.EEG_SIGNAL_FRAME_SIZE = input_shape[1]

            if data.shape[0] != channels:
                return

            data_filtered = filter_eeg_data(data)
            data_clipped = clip_eeg_data(data_filtered)
            data_normalized = normalize_eeg_data(data_clipped)
            data_framed = split_into_frames(np.array(data_normalized))

            result = model.predict(data_framed)

        elif data_type == "MRI":
            try:
                data_trimmed = np.reshape(trim_one(data), CNN_INPUT_SHAPE_MRI)
                data_normalized = normalize(data_trimmed)
                img_for_predict = data_normalized.reshape(1, data_normalized.shape[0], data_normalized.shape[1], 1)
                result = model.predict(img_for_predict)
            except Exception as e:
                self.show_alert(f"Error processing MRI data: {e}")
                return

        return result

    def on_finished(self):
        """
        Completes the prediction process, displays the result, and stops the loading animation.
        """
        self.change_btn_state(True)
        self.movie.stop()
        self.show_result()

    def on_error(self, error):
        """
        Displays an error message.

        :param error: The error message.
        """
        self.change_btn_state(True)
        self.movie.stop()
        self.show_result()
        self.show_alert(f"Error: {error}")

    """
    UI Updates and State Changes
    """

    def show_loading_animation(self):
        """
        Displays a loading animation.
        """
        self.movie = QMovie(os.path.join(self.MAIN_PATH, 'UI', 'loading.gif'))
        self.ui.resultLabel.setMovie(self.movie)
        self.movie.setScaledSize(QSize(40, 40))
        self.movie.start()

    def change_btn_state(self, state):
        """
        Changes the state of UI buttons.

        :param state: The state (True/False) of the buttons.
        """
        self.ui.predictBtn.setEnabled(state)
        self.ui.showGenerated.setEnabled(state)
        self.ui.switchSceneBtn.setEnabled(state)
        self.ui.loadDataBtn.setEnabled(state)
        self.ui.generateNew.setEnabled(state)
        self.ui.showReal.setEnabled(state)

    def show_result(self):
        """
        Displays the prediction result in the UI.
        """
        if self.predictions is None or len(self.predictions) == 0:
            self.ui.resultLabel.setText("-----------")
            return

        predictions_means = []

        for prediction in self.predictions:
            predictions_means.append(np.mean(prediction))

        result, prob = check_result(np.array(predictions_means))

        self.ui.resultLabel.setText(f"{result} ({prob}%)")

        self.curr_idx_eeg = 0
        self.curr_idx_mri = 0
        self.ui.plotLabelEEG.clear()
        self.ui.plotLabelMRI.clear()

        if self.all_data["EEG"]["data"]:
            self.show_plot(
                self.all_data["EEG"]["data"][0], "EEG",
                self.all_data["EEG"]["names"][self.curr_idx_eeg].split("/")[-1]
            )

        if self.all_data["MRI"]["data"]:
            self.show_plot(
                self.all_data["MRI"]["data"][0][0], "MRI",
                self.all_data["MRI"]["names"][self.curr_idx_mri].split("/")[-1]
            )

    """
    Plotting
    """

    def prepare_and_plot_data(self, data_type, radio_adhd, input_number, dialog):
        """
        Prepares and plots MRI data based on the selected options.

        :param data_type: The type of data ('REAL' or 'GENERATED').
        :param radio_adhd: Radio button for adhd data.
        :param input_number: Line edit for the number of images.
        :param dialog: The dialog object.
        """
        self.ui.btnNextPlane.setEnabled(False)
        self.ui.btnPrevPlane.setEnabled(False)

        self.curr_idx_eeg = 0
        self.curr_idx_mri = 0
        self.curr_idx_channel = 0
        self.curr_idx_plane = 0
        self.all_data = {"EEG": {
            "data": [],
            "names": []
        }, "MRI": {
            "data": [],
            "names": []
        }}

        dialog.close()

        file_path = os.path.join(self.MAIN_PATH,
            "MRI", f"{data_type}_MRI",
            f"{'ADHD' if radio_adhd.isChecked() else 'CONTROL'}_{data_type}.pkl"
        )

        data = read_pickle(file_path)

        input_number = min(int(input_number.text()), 20)
        img_numbers = random.sample(range(len(data)), input_number)

        for img_number in img_numbers:
            try:
                self.all_data["MRI"]["data"].append(
                    [data[img_number], np.zeros(data[img_number].shape), np.zeros(data[img_number].shape)]
                )
            except Exception as e:
                print(f"Failed to display image for index {img_number}: {e}")

        self.show_plot(self.all_data["MRI"]["data"][0][0], "MRI", "")

    def show_plot(self, data, data_type, name=""):
        """
        Displays a plot for EEG or MRI data.

        :param data: The data to display.
        :param data_type: The type of data ('EEG' or 'MRI').
        :param name: The file name.
        """
        if data_type == "EEG":
            self.show_plot_eeg(data, name, self.curr_idx_channel)
        if data_type == "MRI":
            self.show_plot_mri(data, name)

    def show_plot_eeg(self, data, name, channel_number):
        """
        Displays a plot for EEG data.

        :param data: The EEG data to display.
        :param name: The file name.
        :param channel_number: The EEG channel number.
        """
        from EEG.config import FS

        fig = Figure()
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        t = np.arange(0, data[channel_number].shape[0]) / FS
        signal = data[channel_number]

        ax.plot(t, signal, label=f'Channel {channel_number}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Sample values')
        ax.set_title(
            f'Signal plot {name}\nChannel: {electrode_positions[channel_number]}'
        )

        buf = io.BytesIO()
        canvas.print_png(buf)
        qpm = QPixmap()
        qpm.loadFromData(buf.getvalue(), 'PNG')
        self.ui.plotLabelEEG.setPixmap(qpm)

    def show_plot_mri(self, img, name):
        """
        Displays an MRI image.

        :param img: The MRI image to display.
        :param name: The file name.
        """
        fig = Figure()
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.imshow(img, cmap="gray")
        ax.set_title(f'MRI image {name}')

        buf = io.BytesIO()
        canvas.print_png(buf)
        qpm = QPixmap()
        qpm.loadFromData(buf.getvalue(), 'PNG')
        self.ui.plotLabelMRI.setPixmap(qpm)

    """
    Navigation (Next/Previous)
    """

    def show_next_plot_eeg(self):
        """
        Displays the next EEG plot.
        """
        if len(self.all_data["EEG"]["data"]) == 0:
            return

        self.curr_idx_eeg += 1
        self.curr_idx_channel = 0

        if self.curr_idx_eeg > len(self.all_data["EEG"]["data"]) - 1:
            self.curr_idx_eeg = len(self.all_data["EEG"]["data"]) - 1

        self.show_plot(
            self.all_data["EEG"]["data"][self.curr_idx_eeg], "EEG",
            self.all_data["EEG"]["names"][self.curr_idx_eeg].split("/")[-1]
        )

    def show_next_channel(self):
        """
        Displays the next EEG channel.
        """
        if len(self.all_data["EEG"]["data"]) == 0:
            return

        self.curr_idx_channel += 1

        if self.curr_idx_channel > 18:
            self.curr_idx_channel = 18

        self.show_plot(
            self.all_data["EEG"]["data"][self.curr_idx_eeg], "EEG",
            self.all_data["EEG"]["names"][self.curr_idx_eeg].split("/")[-1]
        )

    def show_prev_plot_eeg(self):
        """
        Displays the previous EEG plot.
        """
        if len(self.all_data["EEG"]["data"]) == 0:
            return

        self.curr_idx_eeg -= 1
        self.curr_idx_channel = 0

        if self.curr_idx_eeg < 0:
            self.curr_idx_eeg = 0

        self.show_plot(
            self.all_data["EEG"]["data"][self.curr_idx_eeg], "EEG",
            self.all_data["EEG"]["names"][self.curr_idx_eeg].split("/")[-1]
        )

    def show_prev_channel(self):
        """
        Displays the previous EEG channel.
        """
        if len(self.all_data["EEG"]["data"]) == 0:
            return

        self.curr_idx_channel -= 1

        if self.curr_idx_channel < 0:
            self.curr_idx_channel = 0

        self.show_plot(
            self.all_data["EEG"]["data"][self.curr_idx_eeg], "EEG",
            self.all_data["EEG"]["names"][self.curr_idx_eeg].split("/")[-1])

    def show_next_plot_mri(self):
        """
        Displays the next MRI image.
        """
        if len(self.all_data["MRI"]["data"]) == 0:
            return

        self.curr_idx_mri += 1
        self.curr_idx_plane = 0

        if self.curr_idx_mri > len(self.all_data["MRI"]["data"]) - 1:
            self.curr_idx_mri = len(self.all_data["MRI"]["data"]) - 1

        self.show_plot(
            self.all_data["MRI"]["data"][self.curr_idx_mri][self.curr_idx_plane], "MRI",
            self.all_data["MRI"]["names"][self.curr_idx_mri].split("/")[-1] if self.file_paths is not None else "")

    def show_next_plane(self):
        """
        Displays the next MRI plane.
        """
        if len(self.all_data["MRI"]["data"]) == 0:
            return

        self.curr_idx_plane += 1

        if self.curr_idx_plane > 2:
            self.curr_idx_plane = 2

        self.show_plot(
            self.all_data["MRI"]["data"][self.curr_idx_mri][self.curr_idx_plane], "MRI",
            self.all_data["MRI"]["names"][self.curr_idx_mri].split("/")[-1] if self.file_paths is not None else "")

    def show_prev_plot_mri(self):
        """
        Displays the previous MRI image.
        """
        if len(self.all_data["MRI"]["data"]) == 0:
            return

        self.curr_idx_mri -= 1
        self.curr_idx_plane = 0

        if self.curr_idx_mri < 0:
            self.curr_idx_mri = 0

        self.show_plot(
            self.all_data["MRI"]["data"][self.curr_idx_mri][self.curr_idx_plane], "MRI",
            self.all_data["MRI"]["names"][self.curr_idx_mri].split("/")[-1] if self.file_paths is not None else "")

    def show_prev_plane(self):
        """
        Displays the previous MRI plane.
        """
        if len(self.all_data["MRI"]["data"]) == 0:
            return

        self.curr_idx_plane -= 1

        if self.curr_idx_plane < 0:
            self.curr_idx_plane = 0

        self.show_plot(
            self.all_data["MRI"]["data"][self.curr_idx_mri][self.curr_idx_plane], "MRI",
            self.all_data["MRI"]["names"][self.curr_idx_mri].split("/")[-1] if self.file_paths is not None else "")


"""
Worker Class
"""


class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, controller):
        """
        Initializes the worker to process files in a separate thread.

        :param controller: The doctor view controller.
        """
        super().__init__()
        self.controller = controller

    def run(self):
        """
        Starts file processing and model loading.
        """
        try:
            self.load_models()
            self.process_files()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def load_models(self):
        """
        Loads models in the controller.
        """
        self.controller.load_models()

    def process_files(self):
        """
        Processes files in the controller.
        """
        self.controller.process_files()
