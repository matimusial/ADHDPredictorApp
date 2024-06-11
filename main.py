import os
import sys
import traceback

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication
from MRI.main_MRI import MRI
from EEG.main_EEG import EEG
from CONTROLLERS.DoctorViewController import DoctorViewController
from CONTROLLERS.admin_eeg_cnn_agent import AdminEegCnn
from CONTROLLERS.admin_mri_cnn_agent import AdminMriCnn
from CONTROLLERS.admin_mri_gan_agent import AdminMriGan
from CONTROLLERS.generateNew import GenerateNew
from CONTROLLERS.admin_db_view import AdminDbView

def get_base_path():
    """
    Returns:
        str: The base path of the application.
    """
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
current_dir = get_base_path()
UI_PATH = os.path.join(current_dir, 'UI')

print("chuj")
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.av = None
        self.gn = None
        try:
            self.viewController = DoctorViewController(self, UI_PATH, current_dir)
            self.load_doctor_ui()
            self.show()
        except Exception as e:
            print(f"An error occurred during MainWindow initialization: {e}")
            traceback.print_exc()

    def load_admin_db_view(self):
        """
        Loads the admin database view UI and sets up event handlers for the buttons.
        """
        ui_path = os.path.join(current_dir ,"UI", "admin_db.ui")
        ui = uic.loadUi(ui_path, self)
        self.av = AdminDbView(ui)

        ui.deleteBtn.clicked.connect(self.av.show_dialog)
        ui.backBtn.clicked.connect(self.loadAdminEegCnn)

    def load_gen_view(self):
        """
        Load the generator view UI and connect signals to slots.
        """
        ui_path = os.path.join(current_dir, "UI", "genView.ui")
        ui = uic.loadUi(ui_path, self)
        self.gn = GenerateNew(ui)

        ui.backBtn.clicked.connect(self.load_doctor_ui)
        ui.adhdGenInfo.clicked.connect(lambda: self.gn.show_info("adhd"))
        ui.controlGenInfo.clicked.connect(lambda: self.gn.show_info("control"))
        ui.genBtn.clicked.connect(self.gn.generate)
        ui.btnPrevPlot.clicked.connect(self.gn.show_prev_plot_mri)
        ui.btnNextPlot.clicked.connect(self.gn.show_next_plot_mri)
        ui.saveBtn.clicked.connect(self.gn.save_image)

    def load_doctor_ui(self):
        """
        Load the doctor view UI.
        """
        self.viewController = DoctorViewController(self, UI_PATH, current_dir)
        self.viewController.ui.switchSceneBtn.clicked.connect(self.loadAdminEegCnn)
        self.viewController.ui.generateNew.clicked.connect(self.load_gen_view)

    def loadAdminEegCnn(self):
        try:
            self.viewController = AdminEegCnn(self, UI_PATH, current_dir)
            self.viewController.ui.switchSceneBtn.clicked.connect(self.load_doctor_ui)
            self.viewController.ui.CNN_MRI_Button.clicked.connect(self.loadAdminMriCnn)
            self.viewController.ui.GAN_MRI_Button.clicked.connect(self.loadAdminMriGan)
            self.viewController.ui.dbButton.clicked.connect(self.load_admin_db_view)
        except Exception as e:
            print(f"An error occurred while loading Admin EEG CNN: {e}")
            traceback.print_exc()

    def loadAdminMriCnn(self):
        try:
            self.viewController = AdminMriCnn(self, UI_PATH, current_dir)
            self.viewController.ui.switchSceneBtn.clicked.connect(self.load_doctor_ui)
            self.viewController.ui.GAN_MRI_Button.clicked.connect(self.loadAdminMriGan)
            self.viewController.ui.CNN_EEG_Button.clicked.connect(self.loadAdminEegCnn)
            self.viewController.ui.dbButton_2.clicked.connect(self.load_admin_db_view)
        except Exception as e:
            print(f"An error occurred while loading Admin MRI CNN: {e}")
            traceback.print_exc()

    def loadAdminMriGan(self):
        try:
            self.viewController = AdminMriGan(self, UI_PATH, current_dir)
            self.viewController.ui.switchSceneBtn.clicked.connect(self.load_doctor_ui)
            self.viewController.ui.CNN_MRI_Button.clicked.connect(self.loadAdminMriCnn)
            self.viewController.ui.CNN_EEG_Button.clicked.connect(self.loadAdminEegCnn)
            self.viewController.ui.dbButton.clicked.connect(self.load_admin_db_view)
        except Exception as e:
            print(f"An error occurred while loading Admin MRI GAN: {e}")
            traceback.print_exc()

    def run_app(self):
        while True:
            try:
                choice = input('Choose an option:   1-(EEG)   2-(MRI): ')
                if choice not in ['1', '2']:
                    print("Invalid choice. Enter 1 or 2.")
                    continue
                if choice == '1':
                    EEG()
                elif choice == '2':
                    MRI()
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()


app = QApplication(sys.argv)
window = MainWindow()
app.exec_()
