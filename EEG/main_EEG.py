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
        main_choice = input('Choose an option:   1-(run CNN training)   2-(run CNN prediction): ')

        if main_choice == '1':
            save = input('Choose an option:   1-(save model)   2-(do not save model): ')
            if save not in ['1', '2']:
                print("Invalid choice. Enter 1 or 2.")
                continue
            save_model = True if save == '1' else False
            train_cnn_eeg(save_model, TRAIN_PATH, PREDICT_PATH, MODEL_PATH)
            break

        elif main_choice == '2':
            predict(MODEL_CNN_NAME, MODEL_PATH, PREDICT_PATH)
            break

        else:
            print("Invalid choice. Enter 1 or 2.")
