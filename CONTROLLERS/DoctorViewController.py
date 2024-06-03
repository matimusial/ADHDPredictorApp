import os
import io
import numpy as np
import nibabel as nib
import pyedflib
import concurrent.futures
from PyQt5 import uic
from PyQt5.QtCore import QStringListModel, QModelIndex
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QStandardItem, QStandardItemModel
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.io import loadmat
from pandas import read_csv
from keras.models import load_model

from CONTROLLERS.DBConnector import DBConnector
from EEG.config import FS
from EEG.data_preprocessing import filter_eeg_data, clip_eeg_data, normalize_eeg_data
from EEG.file_io import split_into_frames
from EEG.PREDICT.predict import check_result
from MRI.image_preprocessing import trim_one, normalize
from MRI.config import CNN_INPUT_SHAPE_MRI
from EEG.config import CNN_INPUT_SHAPE
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QLineEdit, QLabel, QPushButton, QMessageBox


current_dir = os.path.dirname(__file__)
UI_PATH = os.path.join(current_dir, 'UI')
parent_directory = os.path.dirname(current_dir)
FILE_TYPES = ["mat", "csv", 'edf', 'nii.gz', 'nii']

class DoctorViewController:
    def __init__(self, mainWindow):

        self.mainWindow = mainWindow
        self.ui = uic.loadUi(os.path.join(parent_directory, 'UI', 'doctorView1.ui'), mainWindow)
        self.addEvents()

        # self.db_conn.insert_data_into_models("0.9307", os.path.join('EEG','MODELS','0.9307.keras'), 19, CNN_INPUT_SHAPE, 'cnn_eeg',128,"","")

        self.db_conn = None
        self.filePaths = None
        self.modelEEG = None
        self.modelMRI = None
        self.chosenModelNameEEG = None
        self.chosenModelNameMRI = None
        self.loadedEEGfiles = 0
        self.loadedMRIfiles = 0
        self.currIdxEEG = 0
        self.currIdxMRI = 0
        self.predictions = None
        self.allData = {"EEG": [], "MRI": []}

    def addEvents(self):

        self.ui.loadDataBtn.clicked.connect(self.getFilePaths)

        self.ui.btnNextPlot.clicked.connect(self.showNextPlotEEG)
        self.ui.btnPrevPlot.clicked.connect(self.showPrevPlotEEG)

        self.ui.btnNextPlot_2.clicked.connect(self.showNextPlotMRI)
        self.ui.btnPrevPlot_2.clicked.connect(self.showPrevPlotMRI)

        self.ui.predictBtn.clicked.connect(self.on_pred_click)

        self.ui.showGenerated.clicked.connect(self.showGenerated)


    def showGenerated(self):
        dialog = QDialog(self.ui)
        dialog.setWindowTitle('Wybor Opcji')

        layout = QVBoxLayout()

        radio_healthy = QRadioButton('Zdrowy')
        radio_sick = QRadioButton('Chory')

        layout.addWidget(radio_healthy)
        layout.addWidget(radio_sick)

        label = QLabel('Liczba zdjęć:')
        input_number = QLineEdit()

        layout.addWidget(label)
        layout.addWidget(input_number)

        submit_button = QPushButton('Submit')
        submit_button.clicked.connect(lambda: self.plotGenerated(radio_healthy, radio_sick, input_number, dialog))

        layout.addWidget(submit_button)

        dialog.setLayout(layout)
        dialog.exec_()


    def plotGenerated(self, radio_healthy, radio_sick, input_number, dialog):

        path = f"../MRI/GENERATED_MRI/{radio_healthy}"


    def on_pred_click(self):
        print("Button clicked, starting task...")
        self.runTask()

    def runTask(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.predict)
            future.add_done_callback(self.showResult)

    def predict(self):
        if self.filePaths is None or (self.chosenModelNameEEG is None and self.chosenModelNameMRI is None):
            print("Brak załadowanych plików lub modelu")
            return

        self.loadModels()
        self.processFiles()

    def getFilePaths(self):
        options = QFileDialog.Options()
        fileFilter = ";;".join([f"{ext} files (*.{ext})" for ext in FILE_TYPES])
        defaultPath = os.path.join('CONTROLLERS', 'INPUT_DATA')
        self.filePaths, _ = QFileDialog.getOpenFileNames(self.mainWindow, "Choose files", defaultPath, "", options=options)

        self.loadedEEGfiles = 0
        self.loadedMRIfiles = 0

        for path in self.filePaths:
            if path.endswith('.mat') or path.endswith('.edf'):
                self.loadedEEGfiles += 1
            if path.endswith('.nii') or path.endswith('.nii.gz'):
                self.loadedMRIfiles += 1

        self.ui.dataName.setText(f"{self.loadedEEGfiles} EEG and {self.loadedMRIfiles} MRI files chosen")

        self.getModelNames()

    def getModelNames(self):
        self.chosenModelNameEEG = None
        self.chosenModelNameMRI = None
        self.ui.chosenModelEEG.setText("----------")
        self.ui.chosenModelMRI.setText("----------")

        self.db_conn = DBConnector()
        self.db_conn.establish_connection()
        print(self.db_conn.connection)
        if self.db_conn.connection == None: return


        modelEEG = self.ui.modelListViewEEG.model()
        if modelEEG:
            modelEEG.clear()
        else:
            modelEEG = QStandardItemModel()

        if self.loadedEEGfiles > 0:
            def chooseModelEEG(index: QModelIndex):
                item = modelEEG.itemFromIndex(index)
                self.chosenModelNameEEG = item.text()
                self.ui.chosenModelEEG.setText(self.chosenModelNameEEG)

            modelsList = self.db_conn.select_model_name("type='cnn_eeg'")

            for modelName in modelsList:
                item = QStandardItem(modelName[0])
                item.setEditable(False)
                modelEEG.appendRow(item)


            self.ui.modelListViewEEG.setModel(modelEEG)
            self.ui.modelListViewEEG.doubleClicked.connect(chooseModelEEG)


        modelMRI = self.ui.modelListViewMRI.model()
        if modelMRI:
            modelMRI.clear()
        else:
            modelMRI = QStandardItemModel()

        if self.loadedMRIfiles:
            def chooseModelMRI(index: QModelIndex):
                item = modelMRI.itemFromIndex(index)
                self.chosenModelNameMRI = item.text()
                self.ui.chosenModelMRI.setText(self.chosenModelNameMRI)

            modelsList = self.db_conn.select_model_name("type='cnn_mri'")

            for modelName in modelsList:
                item = QStandardItem(modelName[0])
                item.setEditable(False)
                modelMRI.appendRow(item)

            self.ui.modelListViewMRI.setModel(modelMRI)
            self.ui.modelListViewMRI.doubleClicked.connect(chooseModelMRI)

    def processFiles(self):
        self.predictions = []
        self.allData = {"EEG": [], "MRI": []}

        for path in self.filePaths:
            data = np.array([])
            dataType = ""
            modelName = ""

            if path.endswith('.edf'):
                print("EDF")
                f = pyedflib.EdfReader(path)
                n_channels = f.signals_in_file
                n_samples = f.getNSamples()[0]
                data = np.zeros((n_channels, n_samples))
                for i in range(n_channels):
                    data[i, :] = f.readSignal(i)
                f.close()
                dataType = "EEG"
                model = self.modelEEG

            if path.endswith('.mat'):
                print("MAT")
                file = loadmat(path)
                data_key = list(file.keys())[-1]
                data = file[data_key].T
                dataType = "EEG"
                model = self.modelEEG

            if path.endswith('.csv'):
                print("CSV")
                data = read_csv(path).T
                dataType = "EEG"
                model = self.modelEEG

            if path.endswith('.nii.gz') or path.endswith('.nii'):
                print('NII')
                file = nib.load(path)
                fileData = file.get_fdata()
                (x, y, z, t) = fileData.shape

                frontalPlane = fileData[:, int(y/2), :, int(t/2)]       # widok mózgu z przodu
                sagittalPlane = fileData[int(x/2), :, :, int(t/2)]      # widok mózgu z boku
                horizontalPlane = fileData[:, :, int(z/2), int(t/2)]    # widok mózgu z góry

                data = horizontalPlane
                dataType = "MRI"
                model = self.modelMRI
                print(data.shape)

            result = self.processData(data, model, dataType=dataType)
            self.allData[dataType].append(data)
            self.predictions.append(result)

    def loadModels(self):
        self.modelEEG = None
        self.modelMRI = None

        if self.chosenModelNameEEG is not None:
            self.modelEEG = self.db_conn.select_model(self.chosenModelNameEEG)
        if self.chosenModelNameMRI is not None:
            self.modelMRI = self.db_conn.select_model(self.chosenModelNameMRI)

    def processData(self, DATA, model, dataType="EEG"):
        result = []

        if dataType == "EEG":
            try:
                DATA_FILTERED = filter_eeg_data(DATA)

                DATA_CLIPPED = clip_eeg_data(DATA_FILTERED)

                DATA_NORMALIZED = normalize_eeg_data(DATA_CLIPPED)

                DATA_FRAMED = split_into_frames(np.array(DATA_NORMALIZED))

                result = model.predict(DATA_FRAMED)

            except Exception as e:
                print(f"Error processing EEG data: {e}")
                return

        if dataType == "MRI":
            try:
                DATA_TRIMMED = np.reshape(trim_one(DATA), CNN_INPUT_SHAPE_MRI)

                DATA_NORMALIZED = normalize(DATA_TRIMMED)

                img_for_predict = DATA_NORMALIZED.reshape(1, DATA_NORMALIZED.shape[0], DATA_NORMALIZED.shape[1], 1)

                result = model.predict(img_for_predict)

            except Exception as e:
                print(f"Error processing MRI data: {e}")
                return

        return result

    def showResult(self, future):
        print("Async task finished...")
        if self.predictions is None: return

        predictions_means = []

        for prediction in self.predictions:
            predictions_means.append(np.mean(prediction))

        result, prob = check_result(predictions_means)

        self.ui.resultLabel.setText(f"{result} ({prob}%)")

        self.currIdxEEG = 0
        self.currIdxMRI = 0
        self.ui.plotLabelEEG.clear()
        self.ui.plotLabelMRI.clear()

        if self.allData["EEG"]: self.showPlot(self.allData["EEG"][0], "EEG", "EEG")
        if self.allData["MRI"]: self.showPlot(self.allData["MRI"][0], "MRI", "MRI")

    def showNextPlotEEG(self):
        if(len(self.allData["EEG"]) == 0): return

        self.currIdxEEG += 1

        if self.currIdxEEG > len(self.allData["EEG"])-1:
            self.currIdxEEG = len(self.allData["EEG"])-1

        self.showPlot(self.allData["EEG"][self.currIdxEEG], "EEG", self.filePaths[self.currIdxEEG].split("/")[-1])

    def showPrevPlotEEG(self):
        if(len(self.allData["EEG"]) == 0): return

        self.currIdxEEG -= 1

        if self.currIdxEEG < 0:
            self.currIdxEEG = 0

        self.showPlot(self.allData["EEG"][self.currIdxEEG], "EEG", self.filePaths[self.currIdxEEG].split("/")[-1])
    def showNextPlotMRI(self):
        if len(self.allData["MRI"]) == 0: return

        self.currIdxMRI += 1

        if self.currIdxMRI > len(self.allData["MRI"])-1:
            self.currIdxMRI = len(self.allData["MRI"])-1

        self.showPlot(self.allData["MRI"][self.currIdxMRI], "MRI", self.filePaths[self.currIdxMRI].split("/")[-1])

    def showPrevPlotMRI(self):
        if len(self.allData["MRI"]) == 0: return

        self.currIdxMRI -= 1

        if self.currIdxMRI < 0:
            self.currIdxMRI = 0

        self.showPlot(self.allData["MRI"][self.currIdxMRI], "MRI", self.filePaths[self.currIdxMRI].split("/")[-1])
    def showPlot(self, data, dataType, name):
        if dataType == "EEG":
            self.show_plot_eeg(data, name)
        if dataType == "MRI":
            self.show_plot_mri(data, name)

    def show_plot_eeg(self, data, name, channel_number=5):
        fig = Figure(figsize=(5,5))
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        t = np.arange(0, data[channel_number].shape[0]) / FS
        signal = data[channel_number]

        ax.plot(t, signal, label=f'Kanał {channel_number}')
        ax.set_xlabel('Czas (s)')
        ax.set_ylabel('Wartości próbek')
        ax.set_title(f'Wykres sygnału {name}')
        #ax.legend()

        buf = io.BytesIO()
        canvas.print_png(buf)
        qpm = QPixmap()
        qpm.loadFromData(buf.getvalue(), 'PNG')
        self.ui.plotLabelEEG.setPixmap(qpm)

    def show_plot_mri(self, img, name):

        fig = Figure(figsize=(5,5))
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.imshow(img)
        ax.set_title(f'Zdjęcie mri {name}')
        #ax.colorbar()
        #ax.legend()

        buf = io.BytesIO()
        canvas.print_png(buf)
        qpm = QPixmap()
        qpm.loadFromData(buf.getvalue(), 'PNG')
        self.ui.plotLabelMRI.setPixmap(qpm)


# 1. Wprowadzenie danych

    # obsługa wielu plików naraz (np. lekarz wrzuca naraz 15 eeg i 10 mri)

    # obsługa mri i eeg naraz

    # obsługa eeg o różnych ilościach kanałów
        # na tej podstawie wybranie modelu dostosowanego do konkretnej ilosci kanałów

    # obsługa różnych płaszczyzn mri (różne modele nauczone na różnych płaszczyznach)
    #liczba kanalow; czy eeg czy mri; czestotliwosc probkowania; charakterystyka grupy uczacej modelu
# 2. Wyświetlenie diagnozy

    # diagnoza od razu dla wszystkich wprowadzonych danych

    # wyswietlenie danych na których podstawie zostala postawiona diagnoza (wykresy EEG / zdjecia MRI)
        # dla większej ilości danych możliwość przewijania zdjęć

    # wspolna diagnoza dla roznych danych tego samego pacjenta

# Ponadto xd

    # dodanie do modelu MRI etykiety dotyczącej płaszczyzny mózgu na której uczony był model
    # dodanie do modelu etykiet dotyczących charakterystyki grupy uczącej (np. wiek, płeć, dominująca ręka)
