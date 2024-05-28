import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from EEG.TRAIN.train import train_cnn_eeg
from EEG.PREDICT.predict import predict
from EEG.config import MODEL_CNN_NAME

current_dir = os.path.dirname(__file__)


MODEL_PATH = os.path.join(current_dir, "MODELS")

TRAIN_PATH = os.path.join(current_dir, "TRAIN", "TRAIN_DATA")
PREDICT_PATH = os.path.join(current_dir, "PREDICT", "PREDICT_DATA")


def EEG():
    print("EEG")
    while True:
        main_choice = input('Wybierz opcję:   1-(uruchom trening CNN)   2-(uruchom predict CNN): ')

        if main_choice == '1':
            save = input('Wybierz opcję:   1-(zapisz model)   2-(nie zapisuj modelu): ')
            if save not in ['1', '2']:
                print("Niepoprawny wybór. Wprowadź 1 lub 2.")
                continue
            save_model = True if save == '1' else False
            train_cnn_eeg(save_model, TRAIN_PATH, PREDICT_PATH, MODEL_PATH)
            break

        elif main_choice == '2':
            predict(MODEL_CNN_NAME, MODEL_PATH, PREDICT_PATH)
            break

        else:
            print("Niepoprawny wybór. Wprowadź 1 lub 2.")

#EEG()