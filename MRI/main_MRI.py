import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from MRI.CNN.train import train_cnn
from MRI.CNN.predict import predict_cnn
from MRI.GAN.train import train_gan
from MRI.GAN.show_generated import show_generated
from MRI.GAN.generate import generate_images

current_dir = os.path.dirname(__file__)

REAL_MRI_PATH = os.path.join(current_dir, 'REAL_MRI')

ADHD_GEN_PATH = os.path.join(current_dir, 'GENERATED_MRI', 'ADHD_GENERATED')
CONTROL_GEN_PATH = os.path.join(current_dir, 'GENERATED_MRI', 'CONTROL_GENERATED')

CNN_PREDICT_PATH = os.path.join(current_dir, 'CNN', 'PREDICT_DATA')
CNN_MODEL_PATH = os.path.join(current_dir, 'CNN', 'MODELS')

GAN_MODEL_PATH = os.path.join(current_dir, 'GAN', 'MODELS')


def MRI():
    print("MRI")
    from MRI.config import MODEL_CNN_NAME
    while True:
        main_choice = input('Choose an option:   1-(CNN)   2-(GAN): ')

        if main_choice == '1':
            cnn_choice = input('Choose an option:   1-(run CNN training)   2-(run CNN prediction): ')

            if cnn_choice == '1':
                save = input('Choose an option:   1-(save model)   2-(do not save model): ')
                if save not in ['1', '2']:
                    print("Invalid choice. Enter 1 or 2.")
                    continue
                save_model = True if save == '1' else False
                train_cnn(save_model, REAL_MRI_PATH, CNN_PREDICT_PATH, CNN_MODEL_PATH)
                break

            elif cnn_choice == '2':
                predict_cnn(MODEL_CNN_NAME, CNN_MODEL_PATH, CNN_PREDICT_PATH)
                break

            else:
                print("Invalid choice. Enter 1 or 2.")

        elif main_choice == '2':
            gan_choice = input('Choose an option:   1-(train GAN)   2-(display generated images)   3-(generate images): ')

            if gan_choice == '1':
                save = input('Choose an option:   1-(overwrite model)   2-(do not save model): ')
                if save not in ['1', '2']:
                    print("Invalid choice. Enter 1 or 2.")
                    continue
                save_model = True if save == '1' else False
                for data_type in ["ADHD", "CONTROL"]:
                    train_gan(save_model, data_type, REAL_MRI_PATH, GAN_MODEL_PATH)
                break

            elif gan_choice == '2':
                try:
                    im_amount = int(input('Enter the number of images to display (max 20): '))
                    if im_amount <= 0:
                        print("The number of images must be greater than zero.")
                        continue
                except ValueError:
                    print("Invalid number format. Enter an integer.")
                    continue
                show_generated(im_amount, ADHD_GEN_PATH, CONTROL_GEN_PATH)
                break

            elif gan_choice == '3':
                save = input('Choose an option:   1-(overwrite file)   2-(do not save file): ')
                if save not in ['1', '2']:
                    print("Invalid choice. Enter 1 or 2.")
                    continue
                save_model = True if save == '1' else False
                try:
                    im_amount = int(input('Enter the number of images to generate: '))
                    if im_amount <= 0:
                        print("The number of images must be greater than zero.")
                        continue
                    generate_images(save_model, im_amount, GAN_MODEL_PATH)
                except ValueError:
                    print("Invalid number format. Enter an integer.")
                break

            else:
                print("Invalid choice. Enter 1, 2, or 3.")

        else:
            print("Invalid choice. Enter 1 or 2.")
