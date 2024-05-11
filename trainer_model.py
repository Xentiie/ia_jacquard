import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import traceback

try:
    from alive_progress import alive_it
except ImportError:
    def alive_it(it):
        return it

# Modules personnalisé
from img_definition import *
from neural_network import *

def save(file_name:str, nn:PuzzleNet):
    if savejson == True:
        dict_out = {}
        for name, param in nn.named_parameters():
            if param.requires_grad:
                dict_out[name] = param.data.tolist()
        with open(file_name, 'w+') as f:
            json.dump(dict_out, f)

########################################################################
# -------------------- Configuration du programme -------------------- #
########################################################################

# Configuration initiale
batch_size = 16                                                 # Taille de chaque lot de données traité à chaque itération

local_image_dir = './img'                                       # Chemin du dossier contenant les images locales
dataset_dir = './data'                                          # Dossier contenant les photos téléchargés
savejson = True                                                 # Definit l'enregistrement des poids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Utilisation du GPU si disponible
torch.autograd.set_detect_anomaly(True)                         # Activation de la détection d'anomalies pour le débogage

# Préparation des transformations pour les images d'entrée
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),                                      # Convertit les images en torch.Tensor
    transforms.Normalize((img_mean, img_mean, img_mean), (img_std, img_std, img_std))  # Normalisation des images
])

# Chargement du dataset d'images local OU CIFAR avec les transformations spécifiées
#train_set = torchvision.datasets.ImageFolder(root=local_image_dir, transform=transform)
train_set = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform)

# Configuration du DataLoader pour automatiser le chargement des données, le mélange et le traitement par lots
sample_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Itération sur les données pour récupérer un batch
dataiter = iter(sample_loader)
images, labels = next(dataiter)

# Fonctions de permutation pour préparer les données à être traitées par le réseau
p_images, perms = permute2x2(images)                            # Application de la permutation 2x2 pour simuler un puzzle
assert(matrix_to_perm(perm_to_matrix(perms)).equal(perms))      # Vérification de la cohérence de la permutation

# Configuration des indices pour la séparation en données d'entraînement et de validation
validation_ratio = 0.1                                          # 10% des données pour la validation
total = len(train_set)
ind = list(range(total))
n_train = int(np.floor((1. - validation_ratio) * total))
train_ind, validation_ind = ind[:n_train], ind[n_train:]

# Création des sous-échantillonneurs pour l'entraînement et la validation
train_subsampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
validation_subsampler = torch.utils.data.sampler.SubsetRandomSampler(validation_ind)

# Configuration des DataLoader pour l'entraînement et la validation
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_subsampler, num_workers=0)
validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=validation_subsampler, num_workers=0)



########################################################################
# --------------------  Définition des fonctions  -------------------- #
########################################################################

# Fonction pour calculer l'exactitude des prédictions
def compute_acc(p_pred, p_true, average=True):
    correct = torch.sum((torch.sum(p_pred == p_true, 1) == 4).float())  # Compte les prédictions correctes
    if average:
        return correct / p_pred.size(0)                         # Moyenne des prédictions correctes
    else:
        return correct

# Fonction pour entraîner le modèle
def train_model(model:PuzzleNet, criterion, optimizer, train_loader, validation_loader, n_epochs=40, save_file_name="none"):
    model.to(device)                                            # Déplace le modèle sur le GPU si disponible
    
    # Initialisation des listes pour l'historique des métriques
    loss_history, val_loss_history, acc_history, val_acc_history = [], [], [], []

    try:
        for epoch in range(n_epochs):
            print(f"Epoch: {epoch}")
            model.train()                                       # Mode entraînement
            running_loss, n_correct_pred, n_samples = 0.0, 0, 0

            for inputs, _ in alive_it(train_loader):
                inputs = inputs.to(device)
                x_in, perms = (v.to(device) for v in permute2x2(inputs))    # Permution appliquée aux données d'entrée
                y_in = perm_to_matrix(perms).to(device)                     # Conversion des permissions en matrice

                n_samples += inputs.size(0)
                optimizer.zero_grad()                           # Réinitialisation des gradients
                outputs = model(x_in)                           # Prédictions du modèle
                loss = criterion(outputs, y_in)                 # Calcul de la perte
                loss.backward()                                 # Rétropropagation
                optimizer.step()                                # Mise à jour des poids

                # Mise à jour des statistiques d'entraînement
                n_correct_pred += compute_acc(matrix_to_perm(outputs), perms, False).item()
                running_loss += loss.item() * x_in.size(0)

            _filename = f"./out/model_weights_epoch_{epoch+1}.json"
            print(f"Saving to: {_filename}")
            save(_filename, model)

            # Enregistrement de l'historique de perte et d'exactitude
            loss_history.append(running_loss / n_samples)
            acc_history.append(n_correct_pred / n_samples)

            # Phase de validation
            model.eval()                                        # Mode évaluation
            running_val_loss, n_correct_val_pred, n_val_samples = 0.0, 0, 0

            for inputs, _ in validation_loader:
                inputs = inputs.to(device)
                x_in, perms = (v.to(device) for v in permute2x2(inputs))
                y_in = perm_to_matrix(perms).to(device)

                n_val_samples += inputs.size(0)
                outputs = model(x_in)
                outputs.to(device)

                val_loss = criterion(outputs, y_in)
                running_val_loss += val_loss.item() * x_in.size(0)
                n_correct_val_pred += compute_acc(matrix_to_perm(outputs), perms, False).item()

            val_loss_history.append(running_val_loss / n_val_samples)
            val_acc_history.append(n_correct_val_pred / n_val_samples)

            # Affichage des statistiques pour chaque époque
            print(f"Epoch {epoch+1:03d}: loss={loss_history[-1]:.4f}, val_loss={val_loss_history[-1]:.4f}, acc={acc_history[-1]:.2%}, val_acc={val_acc_history[-1]:.2%}")

        history = {
            'loss': loss_history,
            'val_loss': val_loss_history,
            'acc': acc_history,
            'val_acc': val_acc_history
        }

    except Exception as e:
        print(traceback.format_exc())
        print(f"An error occurred: {e}")
        history = {
            'loss': [],
            'val_loss': [],
            'acc': [],
            'val_acc': []
        }

    # Sauvegarde finale du modèle si un nom de fichier est spécifié
    if save_file_name is not None:
        torch.save({
            'history': history,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_file_name)
        print(f'History is saved with name : {save_file_name}')

    return history

# Fonction pour tester le modèle et calculer l'exactitude
def test_model(model, test_loader):
    running_acc = 0.
    n = 0
    model.eval()  # Mode évaluation pour le test
    for i, data in enumerate(test_loader, 0):
        inputs, _ = data
        x_in, perms = permute2x2(inputs)
        y_in = perm_to_matrix(perms)
        if is_cuda_available:
            x_in, y_in = Variable(x_in.cuda()), Variable(y_in.cuda())
        else:
            x_in, y_in = Variable(x_in), Variable(y_in)
        pred = model(x_in)
        perms_pred = matrix_to_perm(pred.cpu().data)
        running_acc += compute_acc(perms_pred, perms, False)
        n += x_in.size()[0]
    acc = running_acc / n
    return acc


########################################################################
# --------------------  Début de l'entrainement   -------------------- #
########################################################################

# Configuration initiale du modèle et de l'entraînement
n_epochs = 100
sinkhorn_iter = 5
model = PuzzleNet(sinkhorn_iter=sinkhorn_iter)
is_cuda_available = torch.cuda.is_available();
if is_cuda_available:
    model.cuda()

# Sauvegarde du modele OU non
# save_file_name = 'puzzle_jaquard_e{}_s{}.pk'.format(n_epochs, sinkhorn_iter)
save_file_name = None

# Comptage du nombre de paramètres dans le modèle
n_params = 0
for p in model.parameters():
    n_params += np.prod(p.size())
print(f'# of parameters: {n_params}')

# Sauvegarde initiale des poids du modèle
save("./out/model_weights_init.json", model)

# Définition du critère de perte et de l'optimiseur
criterion = nn.BCELoss()                                        # Perte de cross-entropie binaire
optimizer = optim.Adam(model.parameters())                      # Optimiseur Adam

# Entraînement du modèle
history = train_model(model, criterion, optimizer, train_loader, validation_loader, n_epochs=n_epochs, save_file_name=save_file_name)

# Visualisation des courbes de perte et d'exactitude
plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

plt.figure()
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Calcul des exactitudes d'entraînement, de validation et de test
print('Training accuracy: {}'.format(test_model(model, train_loader)))
print('Validation accuracy: {}'.format(test_model(model, validation_loader)))
print('Test accuracy: {}'.format(test_model(model, sample_loader)))

# Traitement d'images de test et visualisation des images permutées et restaurées
test_data_iter = iter(sample_loader)
test_images, _ = next(test_data_iter)
p_images, perms = permute2x2(test_images)

plt.figure()
imshow(torchvision.utils.make_grid(p_images))
plt.title('Inputs')
plt.show()

model.eval()
if is_cuda_available:
    pred = model(Variable(p_images.cuda()))
else:
    pred = model(Variable(p_images))
perms_pred = matrix_to_perm(pred.cpu().data)

plt.figure()
imshow(torchvision.utils.make_grid(restore_original_image_order(p_images, perms_pred)))
plt.title('Restored')
plt.show()
