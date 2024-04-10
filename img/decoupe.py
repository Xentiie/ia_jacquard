# Importation des bibliothèques nécessaires
from PIL import Image
import glob
import random
import os

# Ouverture de l'image à découper
list_image = glob.glob('./image_brute/*.*')

for i, image in enumerate(list_image):
    img = Image.open(image).resize((180, 180))
    img.save(f'./real_data/image{i}.png')

    # Définition du nombre de divisions par côté
    st = 4

    # Calcul de la taille de chaque carré a partir de l'image globale
    total_width, total_height = img.size
    piece_width = total_width / st
    piece_height = total_height / st

    # Initialisation du compteur pour nommer les images découpées
    c = 1

    # Boucles pour parcourir l'image par carrés
    for x in range(st):  # Parcourt les colonnes
        for y in range(st):  # Parcourt les lignes
            # Découpe de l'image
            im2 = img.crop((piece_width * x,
                            piece_height * y,
                            piece_width * x + piece_width,
                            piece_height * y + piece_height))

            # Enregistrement de chaque carré découpé comme une nouvelle image
            im2.save(f'{c}.png')

            # Incrémentation du compteur pour le nom de la prochaine image
            c += 1

    # # Ouverture de la première image pour obtenir les dimensions d'une pièce
    # piece = Image.open("1.png")
    # piece_width, piece_height = piece.size

    # # Création d'une nouvelle image avec la taille totale du puzzle
    # total_width = piece_width * st
    # total_height = piece_height * st
    new_img = Image.new('RGB', (total_width, total_height))

    # Création d'une liste des noms des fichiers images
    pieces = [f'{i}.png' for i in range(1, st**2 + 1)]

    # Mélange aléatoire de la liste des morceaux
    random.shuffle(pieces)

    # Placement des pièces sur la nouvelle image
    for i2, piece_name in enumerate(pieces):
        piece = Image.open(piece_name)
        # Calcul de la position de la pièce actuelle
        x = int((i2 % st) * piece_width)
        y = int((i2 // st) * piece_height)
        # Collage de la pièce sur la nouvelle image
        new_img.paste(piece, (x, y))

    # Enregistrement de l'image finale
    new_img.save(f'./entrainement_data/puzzle_melange{i}.png')
    
    # Concatenation image
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    get_concat_h(Image.open(f'./real_data/image{i}.png'),
                 Image.open(f'./entrainement_data/puzzle_melange{i}.png')).save(f'./concatain_img/concat_img{i}.png')

    # Suppression des fichiers intermédiaires
    for piece_name in pieces:
        os.remove(piece_name)
