# Importation des bibliothèques nécessaires
from PIL import Image
import glob
import random
import os

# Définition des valeurs constantes

IMAGE_DIMENSIONS=(32, 32)
SLICES=2 # Définition du nombre de divisions par côté
ORIGINAL_IMAGES_DIRECTORY="./image_brute"
SLICED_IMAGES_DIRECTORY="./sliced"
RESIZED_IMAGES_DIRECTORY="./resized"
MERGED_IMAGES_DIRECTORY="./merged"
PUZZLE_IMAGES_DIRECTORY="./puzzle"

# Définition d'une fonction pour sauvegarder une image
def save_image(img: Image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

# Ouverture de l'image à découper
list_image = glob.glob(f'./{ORIGINAL_IMAGES_DIRECTORY}/*.*')
print(f"Found {len(list_image)} images")

for i, image in enumerate(list_image):
    img = Image.open(image).resize(IMAGE_DIMENSIONS)
    save_image(img, f'{RESIZED_IMAGES_DIRECTORY}/image{i}.png')

    # Calcul de la taille de chaque carré a partir de l'image globale
    total_width, total_height = img.size
    piece_width = total_width // SLICES
    piece_height = total_height // SLICES

    # Initialization de la liste de toutes les pieces
    pieces = []

    # Boucles pour parcourir l'image par carrés
    j = 0
    for x in range(SLICES):  # Parcourt les colonnes
        for y in range(SLICES):  # Parcourt les lignes
            # Calculate the exact integer indices for the crop
            left = piece_width * x
            upper = piece_height * y
            right = left + piece_width
            lower = upper + piece_height
            # The crop method is inclusive of the left and upper pixel,
            # and exclusive of the right and lower pixel.
            im2 = img.crop((left, upper, right, lower))
            # Rajout de la piece dans la liste
            pieces.append(im2)
            save_image(im2, f'{PUZZLE_IMAGES_DIRECTORY}/puzzle{i}/{j}.png')
            j += 1

    # Création d'une nouvelle image avec la taille totale du puzzle
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
