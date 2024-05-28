import os
import sys
import traceback
from PyQt5.QtWidgets import QMainWindow, QApplication
from MRI.main_MRI import MRI
from EEG.main_EEG import EEG
from CONTROLLERS.DoctorViewController import DoctorViewController
from CONTROLLERS.admin_eeg_cnn_agent import AdminEegCnn
from CONTROLLERS.admin_mri_cnn_agent import AdminMriCnn
from CONTROLLERS.admin_mri_gan_agent import AdminMriGan



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
current_dir = os.path.dirname(__file__)
UI_PATH = rf'{current_dir}/UI'

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.viewController = DoctorViewController(self)
            self.loadDoctorUI()
            self.show()
        except Exception as e:
            print(f"Wystąpił błąd podczas inicjalizacji MainWindow: {e}")
            traceback.print_exc()

    def run_app(self):
        while True:
            try:
                choice = input('Wybierz opcję:   1-(EEG)   2-(MRI): ')
                if choice not in ['1', '2']:
                    print("Niepoprawny wybór. Wprowadź 1 lub 2.")
                    continue
                if choice == '1':
                    self.runEEG()
                elif choice == '2':
                    self.runMRI()
                break
            except Exception as e:
                print(f"Wystąpił błąd: {e}")
                traceback.print_exc()

    def loadDoctorUI(self):
        try:
            self.viewController = DoctorViewController(self)
            self.viewController.ui.switchSceneBtn.clicked.connect(self.loadAdminEegCnn)
        except Exception as e:
            print(f"Wystąpił błąd podczas ładowania UI doktora: {e}")
            traceback.print_exc()

    def loadAdminEegCnn(self):
        try:
            self.viewController = AdminEegCnn(self)
            self.viewController.ui.switchSceneBtn.clicked.connect(self.loadDoctorUI)
            self.viewController.ui.CNN_MRI_Button.clicked.connect(self.loadAdminMriCnn)
            self.viewController.ui.GAN_MRI_Button.clicked.connect(self.loadAdminMriGan)
        except Exception as e:
            print(f"Wystąpił błąd podczas ładowania Admin EEG CNN: {e}")
            traceback.print_exc()

    def loadAdminMriCnn(self):
        try:
            self.viewController = AdminMriCnn(self)
            self.viewController.ui.switchSceneBtn.clicked.connect(self.loadDoctorUI)
            self.viewController.ui.GAN_MRI_Button.clicked.connect(self.loadAdminMriGan)
            self.viewController.ui.CNN_EEG_Button.clicked.connect(self.loadAdminEegCnn)
        except Exception as e:
            print(f"Wystąpił błąd podczas ładowania Admin MRI CNN: {e}")
            traceback.print_exc()

    def loadAdminMriGan(self):
        try:
            self.viewController = AdminMriGan(self)
            self.viewController.ui.switchSceneBtn.clicked.connect(self.loadDoctorUI)
            self.viewController.ui.CNN_MRI_Button.clicked.connect(self.loadAdminMriCnn)
            self.viewController.ui.CNN_EEG_Button.clicked.connect(self.loadAdminEegCnn)
        except Exception as e:
            print(f"Wystąpił błąd podczas ładowania Admin MRI GAN: {e}")
            traceback.print_exc()

    def runEEG(self):
        try:
            EEG()
        except Exception as e:
            print(f"Wystąpił błąd podczas uruchamiania EEG: {e}")
            traceback.print_exc()

    def runMRI(self):
        try:
            MRI()
        except Exception as e:
            print(f"Wystąpił błąd podczas uruchamiania MRI: {e}")
            traceback.print_exc()


try:
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec_()
    # EEG()
except Exception as e:
    print(f"Wystąpił błąd podczas uruchamiania aplikacji: {e}")
    traceback.print_exc()
