import random
from matplotlib import pyplot as plt

from MRI.file_io import read_pickle


def show_generated(im_amount=3, adhd_path="", control_path=""):
    """
    Displays a specified number of generated images from ADHD and CONTROL generated datasets.

    Parameters:
    im_amount (int, optional): Number of images to display. Default is 3. Maximum is 20.
    ADHD_PATH (str, optional): Path to the ADHD dataset link (without extension).
    CONTROL_PATH (str, optional): Path to the CONTROL dataset link (without extension).

    """
    try:
        ADHD = read_pickle(f"{adhd_path}.pkl")
        CONTROL = read_pickle(f"{control_path}.pkl")
    except Exception as e:
        print(f"Nie udało się odczytać plików z wygenerowanymi danymi: {e}")
        return

    if im_amount >= 20:
        im_amount = 20

    range_list = list(range(len(ADHD)))
    img_numbers = random.sample(range_list, im_amount)

    for i, img_number in enumerate(img_numbers):
        try:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.title(f"ADHD {img_number+1}")
            plt.imshow(ADHD[img_number], cmap="gray")

            plt.subplot(1, 2, 2)
            plt.title(f"CONTROL {img_number+1}")
            plt.imshow(CONTROL[img_number], cmap="gray")

            plt.show()
        except Exception as e:
            print(f"Nie udało się wyświetlić obrazu dla indeksu {img_number}: {e}")
