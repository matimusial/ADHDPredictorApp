import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from MRI.CNN.train import train_cnn
from MRI.CNN.predict import predict_cnn
from MRI.GAN.train import train_gan
from MRI.GAN.show_generated import show_generated
from MRI.GAN.generate import generate_images
from MRI.config import MODEL_CNN_NAME

current_dir = os.path.dirname(__file__)

REAL_MRI_PATH = os.path.join(current_dir, 'REAL_MRI')

ADHD_GEN_PATH = os.path.join(current_dir, 'GENERATED_MRI', 'ADHD_GENERATED')
CONTROL_GEN_PATH = os.path.join(current_dir, 'GENERATED_MRI', 'CONTROL_GENERATED')

CNN_PREDICT_PATH = os.path.join(current_dir, 'CNN', 'PREDICT_DATA')
CNN_MODEL_PATH = os.path.join(current_dir, 'CNN', 'MODELS')

GAN_MODEL_PATH = os.path.join(current_dir, 'GAN', 'MODELS')


def MRI():
    print("MRI")
    while True:
        main_choice = input('Wybierz opcję:   1-(CNN)   2-(GAN): ')

        if main_choice == '1':
            cnn_choice = input('Wybierz opcję:   1-(uruchom trening CNN)   2-(uruchom predict CNN): ')

            if cnn_choice == '1':
                save = input('Wybierz opcję:   1-(zapisz model)   2-(nie zapisuj modelu): ')
                if save not in ['1', '2']:
                    print("Niepoprawny wybór. Wprowadź 1 lub 2.")
                    continue
                save_model = True if save == '1' else False
                train_cnn(save_model, REAL_MRI_PATH, CNN_PREDICT_PATH, CNN_MODEL_PATH)
                break

            elif cnn_choice == '2':
                predict_cnn(MODEL_CNN_NAME, CNN_MODEL_PATH, CNN_PREDICT_PATH)
                break

            else:
                print("Niepoprawny wybór. Wprowadź 1 lub 2.")

        elif main_choice == '2':
            gan_choice = input('Wybierz opcję:   1-(trenuj GAN)   2-(wyświetl wygenerowane zdjęcia)   3-(wygeneruj zdjęcia): ')

            if gan_choice == '1':
                save = input('Wybierz opcję:   1-(nadpisz model)   2-(nie zapisuj modelu): ')
                if save not in ['1', '2']:
                    print("Niepoprawny wybór. Wprowadź 1 lub 2.")
                    continue
                save_model = True if save == '1' else False
                for data_type in ["ADHD", "CONTROL"]:
                    train_gan(save_model, data_type, REAL_MRI_PATH, GAN_MODEL_PATH)
                break

            elif gan_choice == '2':
                try:
                    im_amount = int(input('Podaj ilość zdjęć do wyświetlenia (max 20): '))
                    if im_amount <= 0:
                        print("Liczba zdjęć musi być większa od zera.")
                        continue
                except ValueError:
                    print("Niepoprawny format liczby. Podaj liczbę całkowitą.")
                    continue
                show_generated(im_amount, ADHD_GEN_PATH, CONTROL_GEN_PATH)
                break
            elif gan_choice == '3':
                save = input('Wybierz opcję:   1-(nadpisz plik)   2-(nie zapisuj pliku): ')
                if save not in ['1', '2']:
                    print("Niepoprawny wybór. Wprowadź 1 lub 2.")
                    continue
                save_model = True if save == '1' else False
                try:
                    im_amount = int(input('Podaj ilość zdjęć do wygenerowania: '))
                    if im_amount <= 0:
                        print("Liczba zdjęć musi być większa od zera.")
                        continue
                    generate_images(save_model, im_amount, GAN_MODEL_PATH)
                except ValueError:
                    print("Niepoprawny format liczby. Podaj liczbę całkowitą.")
                break
            else:
                print("Niepoprawny wybór. Wprowadź 1, 2 lub 3.")

        else:
            print("Niepoprawny wybór. Wprowadź 1 lub 2.")

#MRI()