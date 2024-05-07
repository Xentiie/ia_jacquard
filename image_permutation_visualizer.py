import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from img_definition import *

#Configuration
batch_size = 32
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

# Fonctions de permutation
assert(matrix_to_perm(perm_to_matrix(perms)).equal(perms))

# Affiche les images permutées
plt.figure(figsize=(20, 5))
imshow(torchvision.utils.make_grid(p_images))
plt.show()

# Affiche les images restaurés
plt.figure(figsize=(20, 5))
imshow(torchvision.utils.make_grid(restore_original_image_order(p_images, perms)))
plt.show() 