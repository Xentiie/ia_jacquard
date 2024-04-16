# Importation des bibliothèques nécessaires
from PIL import Image
import glob
import random
import os

# Définition des valeurs constantes

IMAGE_DIMENSIONS=(256, 256)
SLICES=4 # Définition du nombre de divisions par côté
ORIGINAL_IMAGES_DIRECTORY="./image_brute"
SLICED_IMAGES_DIRECTORY="./sliced"
RESIZED_IMAGES_DIRECTORY="./resized"
MERGED_IMAGES_DIRECTORY="./merged"

# Définition d'une fonction pour sauvegarder une image
def save_image(img: Image, path):
    if (not os.path.exists(os.path.dirname(path))):
        os.mkdir(os.path.dirname(path))
    img.save(path)

# Ouverture de l'image à découper
list_image = glob.glob(f'./{ORIGINAL_IMAGES_DIRECTORY}/*.*')
print(f"Found {len(list_image)} images")

for i, image in enumerate(list_image):
    img = Image.open(image).resize(IMAGE_DIMENSIONS)
    save_image(img, f'{RESIZED_IMAGES_DIRECTORY}/image{i}.png')

    # Calcul de la taille de chaque carré a partir de l'image globale
    total_width, total_height = img.size
    piece_width = total_width / SLICES
    piece_height = total_height / SLICES

    # Initialization de la liste de toutes les pieces
    pieces = []

    # Boucles pour parcourir l'image par carrés
    for x in range(SLICES):  # Parcourt les colonnes
        for y in range(SLICES):  # Parcourt les lignes
            # Découpe de l'image
            im2 = img.crop((piece_width * x,
                            piece_height * y,
                            piece_width * x + piece_width,
                            piece_height * y + piece_height))
            #Rajout de la piece dans la liste
            pieces.append(im2)

    # # Création d'une nouvelle image avec la taille totale du puzzle
    # total_width = piece_width * st
    # total_height = piece_height * st
    new_img = Image.new('RGB', (total_width, total_height))

    # Mélange aléatoire de la liste des morceaux
    random.shuffle(pieces)

    # Placement des pièces sur la nouvelle image
    for j, piece in enumerate(pieces):
        # Calcul de la position de la pièce actuelle
        x = int((j % SLICES) * piece_width)
        y = int((j // SLICES) * piece_height)
        # Collage de la pièce sur la nouvelle image
        new_img.paste(piece, (x, y))

    # Enregistrement de l'image finale
    save_image(new_img, f'{SLICED_IMAGES_DIRECTORY}/puzzle_melange{i}.png')

    # Concatenation image
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    merged_image = get_concat_h(img, new_img)
    save_image(merged_image, f'{MERGED_IMAGES_DIRECTORY}/merged{i}.png')

