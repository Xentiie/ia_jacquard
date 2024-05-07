import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from img_definition import *

# Configuration
# Un "batch" désigne un ensemble d'échantillons de données 
# sur lesquels le modèle est entraîné ou évalué en une seule itération.
# La division en plusieurs "batch" permet un meilleur ajustement des poids du modéle 
batch_size = 16
local_image_dir = './img'

# Transformations à appliquer sur les images
transform = transforms.Compose(
    [#transforms.RandomHorizontalFlip(),
     #transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((img_mean, img_mean, img_mean), (img_std, img_std, img_std))])

# Charger le dataset local
train_set = torchvision.datasets.ImageFolder(root=local_image_dir, transform=transform)

# DataLoader pour les images locales
sample_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

# Itération sur un batch d'images
dataiter = iter(sample_loader)
images, labels = next(dataiter)

# permutation des tuiles
p_images, perms = permute2x2(images)

# Vos fonctions de permutation ici
assert(matrix_to_perm(perm_to_matrix(perms)).equal(perms))


# Prepare entrainement, validation et exemple de test
# Definition du ratio de validation
validation_ratio = 0.18

# Calcul des indices pour la base d'entrainement et celle de validation
total = len(train_set)                                      # Calcul le nombre total d'échantillons dans l'ensemble d'entrainement
ind = list(range(total))                                    # Crée une liste d'indice
n_train = int(np.floor((1. - validation_ratio) * total))    # Calcule le nombre d'échantillons qui seront utilisés pour l'entraînement, dans ce cas 82%
train_ind, validation_ind = ind[:n_train], ind[n_train:]    # Sépare les indices en deux groupes, les premiers 82% pour l'entraînement et les 18% restants pour la validation.

# Création des échantilloneurs à partir des indices spécifiés pour l'entraînement et la validation.
train_subsampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
validation_subsampler = torch.utils.data.sampler.SubsetRandomSampler(validation_ind)

# Création des DataLoader pour l'entraînement et la validation
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_subsampler, num_workers=0)
validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                sampler=validation_subsampler, num_workers=0)
# Le DataLoader utilise un échantillonneur (Sampler) pour savoir quelles données charger et dans quel ordre.
# DataLoader : Gère le chargement, le mélange, le batching, et le chargement parallèle des données.
# Sampler : Définit l'ordre dans lequel les échantillons sont tirés du dataset.

for i, (images, labels) in enumerate(train_loader):
    print(f"Train Batch {i + 1}:")
    print(f"  Number of images: {images.size(0)}")  # images.size(0) donne le nombre d'images dans le batch d'énumération

for i, (images, labels) in enumerate(validation_loader):
    print(f"Validation Batch {i + 1}:")
    print(f"  Number of images: {images.size(0)}")  # images.size(0) donne le nombre d'images dans le batch de validation
