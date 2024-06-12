import os
from PIL import Image

# Ścieżka do folderu z obrazami
folder_path = (''
               ''
               '')  # Zamień na faktyczną ścieżkę do folderu

# Lista plików w folderze
file_list = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

# Posortowanie plików (opcjonalnie, jeśli chcesz określoną kolejność)
file_list.sort()

# Wczytanie obrazów
images = [Image.open(os.path.join(folder_path, file)) for file in file_list]

# Stworzenie GIF-a
output_path = 'ścieżka/do/zapisu/twojego_gif.gif'  # Zamień na faktyczną ścieżkę do zapisu GIF-a
images[0].save(output_path, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
