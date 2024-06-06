import os
import numpy as np
from tensorflow.keras.models import load_model

from MRI.plot_mri import plot_mri
from MRI.file_io import save_pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generate_images(save, im_amount, model_path):
    """
    Generates MRI images using a trained GAN model.

    Args:
        save (bool): Whether to save the generated images.
        im_amount (int): The number of images to generate.
        model_path (str): The path to the GAN model.

    """
    from MRI.config import GENERATE_GAN_DISP_INTERVAL
    for data_type in ["ADHD", "CONTROL"]:
        try:
            generator = load_model(os.path.join(model_path, f'{data_type}_GAN.keras'))
        except Exception as e:
            print(f"Failed to load the GAN model: {e}")
            print("Did you download the models from the link in the folder MRI/GAN/MODELS?")
            return

        data = []
        for i in range(im_amount):
            noise = np.random.normal(0, 1, [1, 100])
            try:
                generated_image = generator.predict(noise)
            except Exception as e:
                print(f"Failed to generate image: {e}")
                return
            generated_image = generated_image * 0.5 + 0.5
            data.append(generated_image[0])

            if (i + 1) % GENERATE_GAN_DISP_INTERVAL == 0:
                plot_mri(generated_image[0], f"Image index {data_type}: {i + 1}")

        if save:
            save_pickle(os.path.join("GENERATED_MRI", f"{data_type}_GENERATED.pkl"), data)
