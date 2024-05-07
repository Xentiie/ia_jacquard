import torch

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Tableau définit les points de départ de 4 tuiles pour une image de 32x32 pixels, pour effectuer une permutation
tile_start_indices = [(0, 0), (16, 0), (16, 16), (0, 16)]

# Valeur de normalisation des données pour avoir une plage entre -1 et 1
img_mean = 0.5
img_std = 0.5
    
def permute2x2(images):
    """
    Découpe les images en 2x2 = 4 piéces et permuttes aléatoirement les piéces.
    """
    p_images = torch.FloatTensor(images.size())         # Crée un tensor de float (tableau multidimensionnel d’éléments) pour les images permuté, de la même taille que l'original
    perms = torch.LongTensor(images.size()[0], 4)       # Crée un nouveau tensor pour stocker l'indice de chaque piéce permutée
    
    # Boucle de permutation
    for i in range(images.size()[0]):                   # Itère sur chaque image
        p = torch.randperm(4)                           # Crée un vecteur aléatoire de taille 4 pour les nouvelles positions des 4 piéces
        for j in range(4):                              # Itère sur chaque piéce de l'image
            sr, sc = tile_start_indices[j]                       # Récupère les coordonnées du coin supérieur gauche (start row, start column)
            tr, tc = tile_start_indices[p[j]]                    # Utilise la permutation p pour trouver les nouvelles coordonnées du coin supérieur gauche (target row, target column)
            p_images[i, :, tr:tr+16, tc:tc+16] = images[i, :, sr:sr+16, sc:sc+16]   #Copie la pièce de l'image originale à la nouvelle position dans l'image permutée.
        perms[i,:] = p                                  # Enregistre la permutation utilisée pour l'image i
    
    return(p_images, perms)

def restore_original_image_order(p_images, perms):
    """
    Prend un ensemble d'images permutées et les vecteurs de permutation correspondants, puis reconstitue les images originales.
    """
    images = torch.FloatTensor(p_images.size())
    
    # Boucle de reconstitution
    for i in range(images.size()[0]):
        for j in range(4):
            sr, sc = tile_start_indices[j]
            tr, tc = tile_start_indices[perms[i, j]]
            images[i, :, sr:sr+16, sc:sc+16] = p_images[i, :, tr:tr+16, tc:tc+16]
    
    return images

def perm_to_matrix(perms):
    """
    Convertit les vecteurs de permutation en matrices
    """
    n = perms.size()[0]
    mat = torch.zeros(n, 4, 4)                          # Crée une matrice de 0, ou n est le nombre d'image
    
    for i in range(n):                                  # itère sur chaque permutation
        for j in range(4):                              # Pour chaque élément pour la permutation place un 1 dans la matrice à
            mat[i, j, perms[i, j]] = 1.                 # la position correspondante.
    
    return mat.view(n, -1)                              # Transforme la matrice 3d en 2d, plus simple pour la manipulation

def matrix_to_perm(x):
    """
    Convertit les matrices en vecteurs de permutation
    """

    n = x.size()[0]
    x = x.view(n, 4, 4)
    _, ind = x.max(2)                                   # _ (les valeurs maximales, non utilisées), ind (indices de permutation)
    
    return ind
    
def imshow(img, title=None):
    """
    Affiche une image.
    """
    img = img * img_std + img_mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title != None:
        plt.title(title)