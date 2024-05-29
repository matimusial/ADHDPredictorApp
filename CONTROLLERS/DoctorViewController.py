import os
import io
import numpy as np
import nibabel as nib
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

current_dir = os.path.dirname(__file__)
UI_PATH = os.path.join(current_dir, 'UI')
parent_directory = os.path.dirname(current_dir)
FILE_TYPES = ["mat", "csv", 'edf', 'nii.gz', 'nii']

class DoctorViewController:
    def __init__(self, mainWindow):

        self.mainWindow = mainWindow
        self.ui = uic.loadUi(os.path.join(parent_directory, 'UI', 'doctorView.ui'), mainWindow)
        self.addEvents()

        self.db_conn = DBConnector()
        self.filePaths = None
        self.modelEEG = None
        self.modelMRI = None
        self.chosenModelNameEEG = None
        self.chosenModelNameMRI = None
        self.currIdxEEG = 0
        self.currIdxMRI = 0
        self.predictions = []
        self.allData = {"EEG": [], "MRI": []}


    def addEvents(self):

        self.ui.loadDataBtn.clicked.connect(self.getFilePaths)

        self.ui.btnNextPlot.clicked.connect(self.showNextPlotEEG)
        self.ui.btnPrevPlot.clicked.connect(self.showPrevPlotEEG)

        self.ui.btnNextPlot_2.clicked.connect(self.showNextPlotMRI)
        self.ui.btnPrevPlot_2.clicked.connect(self.showPrevPlotMRI)

        self.ui.predictBtn.clicked.connect(self.predict)

    def predict(self):
        # if self.filePaths is None or (self.chosenModelNameEEG is None and self.chosenModelNameMRI is None):
        #     print("Brak załadowanych plików lub modelu")
        #     return

        #self.loadModels()
        self.processFiles()
        self.showResult()

    def getFilePaths(self):
        options = QFileDialog.Options()
        fileFilter = ";;".join([f"{ext} files (*.{ext})" for ext in FILE_TYPES])
        defaultPath = os.path.join('CONTROLLERS','INPUT_DATA')
        self.filePaths, _ = QFileDialog.getOpenFileNames(self.mainWindow, "Choose files", defaultPath, "", options=options)
        self.getModelNames()

    def getModelNames(self):
        def chooseModelEEG(index: QModelIndex):
            item = modelEEG.itemFromIndex(index)
            self.chosenModelEEG = item.text()
            self.ui.chosenModelEEG.setText(self.chosenModelEEG)
        def chooseModelMRI(index: QModelIndex):
            item = modelMRI.itemFromIndex(index)
            self.chosenModelMRI = item.text()
            self.ui.chosenModelMRI.setText(self.chosenModelMRI)

        modelsList = self.db_conn.select_model_name("type='cnn_eeg'")
        modelEEG = QStandardItemModel()

        for modelName in modelsList:
            item = QStandardItem(modelName[0])
            item.setEditable(False)
            modelEEG.appendRow(item)

        self.ui.modelListViewEEG.setModel(modelEEG)
        self.ui.modelListViewEEG.doubleClicked.connect(chooseModelEEG)

        modelsList = self.db_conn.select_model_name("type='cnn_mri'")
        modelMRI = QStandardItemModel()

        for modelName in modelsList:
            item = QStandardItem(modelName[0])
            item.setEditable(False)
            modelMRI.appendRow(item)

        self.ui.modelListViewMRI.setModel(modelMRI)
        self.ui.modelListViewMRI.doubleClicked.connect(chooseModelMRI)

    def processFiles(self):
        print("")
        for path in self.filePaths:
            data = np.array([])
            dataType = ""
            modelName = ""

            # załaduj plik ( zależnie od typu inaczej, po to tyle if'ów)
            # zdecyduj czy eeg czy mri  (na podstawie struktury/rozszerzenia)
            # wybierz z niego potrzebne dane
            # wybierz model na podstawie struktury danych (np dla EEG różna ilość kanałów, dla mri różna płaszczyzna mózgu)
            # wrzuć dane w model
            # zwróc wynik

            if path.endswith('.edf'):
                print("EDF")
                #file = EdfReader(path)
                #signalsNum = file.signals_in_file
                #print(type(file.readsignal(1)))
                #signals = [file.readsignal(i) for i in range(signalsNum)]

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

            if path.endswith('.mat'):
                print("MAT")
                file = loadmat(path)
                data_key = list(file.keys())[-1]
                data = file[data_key].T
                dataType = "EEG"
                model = self.modelEEG

            if path.endswith('.csv'):
                print("CSV")
                data = read_csv(path)

            model = self.getModel(dataType, "0.9307")
            result = self.processData(data, model, dataType=dataType)
            self.allData[dataType].append(data)
            self.predictions.append(result)

    def getModel(self, modelType, modelName):
        # charakterystyka danych uczących
        if modelType == "EEG":
            model = load_model(os.path.join('EEG', 'MODELS', f'{modelName}.keras'))
        if modelType == "MRI":
            model = load_model(os.path.join('MRI', 'CNN', 'MODELS', f'{modelName}.keras'))

        return model

    def loadModels(self):
        self.modelMRI = None
        self.modelEEG = None

    def processData(self, DATA, model, dataType="EEG"):
        result = []

        if dataType == "EEG":
            DATA_FILTERED = filter_eeg_data(DATA)

            DATA_CLIPPED = clip_eeg_data(DATA_FILTERED)

            DATA_NORMALIZED = normalize_eeg_data(DATA_CLIPPED)

            DATA_FRAMED = split_into_frames(np.array(DATA_NORMALIZED))

            result = model.predict(DATA_FRAMED)

        if dataType == "MRI":
            DATA_TRIMMED = np.reshape(trim_one(DATA), CNN_INPUT_SHAPE_MRI)

            DATA_NORMALIZED = normalize(DATA_TRIMMED)

            img_for_predict = DATA_NORMALIZED.reshape(1, DATA_NORMALIZED.shape[0], DATA_NORMALIZED.shape[1], 1)

            result = model.predict(img_for_predict)

        return result

    def showResult(self):
        predictions_means = []

        for prediction in self.predictions:
            predictions_means.append(np.mean(prediction))

        result, prob = check_result(predictions_means)

        self.ui.resultLabel.setText(f"{result} ({prob}%)")

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
