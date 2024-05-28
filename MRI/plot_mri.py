import matplotlib.pyplot as plt

def plot_mri(img, title=""):
    try:
        plt.figure()
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.show()
    except Exception as e:
        print("Wystąpił błąd podczas wyświetlania obrazu MRI:", str(e))
        return
