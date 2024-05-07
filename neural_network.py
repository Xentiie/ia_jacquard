from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def sinkhorn(A, n_iter=4):
    """
    Fonction sinkhorn
    La fonction sinkhorn implémente les itérations de Sinkhorn, 
    qui sont utilisées pour normaliser une matrice afin qu'elle 
    devienne une matrice de permutation douce (une matrice où 
    chaque ligne et chaque colonne somme à 1). Cette fonction est 
    utilisée pour transformer les sorties du réseau de neurones 
    en une forme qui ressemble plus à une matrice de permutation.

    Processus :

    Pour un nombre donné d'itérations (n_iter), la matrice A est 
    normalisée successivement par ses lignes puis par ses colonnes. 
    Cela force la matrice à approcher les propriétés d'une matrice 
    bistochastique où la somme des éléments de chaque ligne 
    et chaque colonne est égale à 1.
    """
    for i in range(n_iter):
        A = A / A.sum(dim=1, keepdim=True)
        A = A / A.sum(dim=2, keepdim=True)

    return A

class SimpleConvNet(nn.Module):
    """
    Cette classe définit un réseau de neurones convolutionnel simple. 
    
    Trois couches de convulation, conv1, conv2, conv3, pour extraire des 
    caractéristiques spatiales des images. 
    Chaque couche est suivie par une fonction d'activation ReLU.
    
    pool1 - Une couche de pooling pour réduire les dimensions 
    spatiales des caractéristiques, réduisant ainsi la quantité 
    de paramètres et de calculs dans les couches suivantes.
    
    fc1, fc2 - Couches denses pour traiter les caractéristiques extraites 
    et les transformer en un vecteur de caractéristiques plus abstrait.
    """
    def __init__(self):
        super().__init__()
        
        # Définit une couche de convolution qui prend en entrée des images
        # Premier paramètre (3) : Il représente le nombre de canaux en entrée de la couche.
        
        # Deuxième paramètre (8) : Ce chiffre indique le nombre de filtres (ou noyaux de convolution) 
        # que la couche va utiliser. Chaque filtre détecte des caractéristiques spécifiques dans l'image, 
        # comme les bords, les textures ou d'autres motifs. Avoir plus de filtres permet au réseau de 
        # capturer une gamme plus large de caractéristiques, mais cela augmente aussi la complexité du 
        # modèle et le nombre de paramètres à apprendre.

        # Troisième paramètre (3) : C'est la taille de chaque filtre, ici un filtre de 3x3 pixels. 
        # La taille du filtre détermine l'étendue de l'aire locale sur laquelle le réseau effectue 
        # ses calculs pour produire une seule valeur dans la carte de caractéristiques résultante.
        # 3 x 16 x 16 input
        self.conv1 = nn.Conv2d(3, 8, 3)
        
        # Définit une seconde couche de convolution qui prend les 8 cartes de caractéristiques de la 
        # couche précédente et produit 8 nouvelles cartes, avec des noyaux de 3x3.
        # 8 x 14 x 14
        self.conv2 = nn.Conv2d(8, 8, 3)
        
        # Applique une normalisation par lot sur les 8 cartes de caractéristiques sortant de conv2, 
        # aidant à stabiliser l'apprentissage.
        # La normalisation par lots (Batch Normalization), est utilisée pour améliorer la vitesse, 
        # la performance et la stabilité de l'entraînement des réseaux de neurones. Elle normalise 
        # la sortie de chaque couche de convolution en soustrayant la moyenne du lot et en 
        # divisant par l'écart-type du lot.
        self.conv2_bn = nn.BatchNorm2d(8)
        
        # Applique un max pooling avec une fenêtre de 2x2, ce qui réduit les dimensions 
        # spatiales de moitié (de 12x12 à 6x6)
        # 8 x 12 x 12
        # Le pooling réduit les dimensions spatiales des cartes de caractéristiques. 
        # Cela permet de diminuer la quantité de calculs et de paramètres dans le réseau, 
        # ce qui aide à contrôler le surapprentissage. Le pooling le plus courant est le max 
        # pooling, qui renvoie la valeur maximale d'une région de la carte de caractéristiques.
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Troisième couche de convolution, produisant 16 cartes de caractéristiques à partir des 8 entrantes,
        # avec des noyaux de 3x3.
        # 8 x 6 x 6
        self.conv3 = nn.Conv2d(8, 16, 3)
        # Normalisation par lot pour les 16 cartes de caractéristiques sortant de conv3.
        self.conv3_bn = nn.BatchNorm2d(16)
        
        # Couche totalement connectée (dense) qui transforme les données de la forme aplaties 
        # (16 cartes de 4x4) en un vecteur de 128 dimensions.
        # 16 x 4 x 4
        # Une couche entièrement connectée (fully connected layer ou dense layer) est une couche où 
        # chaque neurone est connecté à tous les neurones de la couche précédente. Ces couches sont g
        # énéralement placées vers la fin d'un CNN après les couches convolutionnelles et de pooling. 
        # Elles sont utilisées pour combiner les caractéristiques (qui ont été extraites à travers le réseau) 
        # en prédictions finales pour la tâche de classification ou de régression. 
        # Les couches entièrement connectées transforment les cartes de caractéristiques réduites en un 
        # vecteur de prédictions ou de scores pour chaque classe possible.
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        # Normalisation par lot pour le vecteur de 128 dimensions.
        self.fc1_bn = nn.BatchNorm1d(128)
        
        # Seconde couche totalement connectée qui prend les 128 dimensions et les transforme en un nouveau vecteur de 128 dimensions.
        # L'ajout de plusieurs couches entièrement connectées augmente la capacité du réseau à apprendre 
        # des relations complexes entre les caractéristiques extraites par les couches convolutionnelles. 
        # Chaque couche entièrement connectée peut être vue comme une étape de transformation des 
        # caractéristiques, où le réseau peut combiner et recombiner les caractéristiques de différentes 
        # manières pour former des représentations de plus en plus abstraites et potentiellement plus informatives.
        self.fc2 = nn.Linear(128, 128)
        # Normalisation par lot pour la sortie de la couche dense.
        self.fc2_bn = nn.BatchNorm1d(128)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2_bn(self.conv2(x)))
        
        x = self.pool1(x)
        
        x = F.relu(self.conv3_bn(self.conv3(x)))
        
        x = x.view(-1, 16 * 4 * 4)
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))

        return x

class PuzzleNet(nn.Module):
    """
    Cette classe définit un réseau de neurones conçu pour résoudre des puzzles 4x4. 
    Il utilise le SimpleConvNet pour traiter chaque pièce du puzzle et 
    combine leurs caractéristiques pour prédire l'arrangement correct des pièces.
    
    Chaque pièce de l'image d'entrée est traitée séparément par le même réseau convolutionnel.
    
    Les caractéristiques extraites de chaque pièce sont concaténées.
    
    Output Layer (fc2): Une couche de sortie qui prédit les positions 
    relatives des pièces sous forme de matrice 3x3.
    
    Activation Sigmoid et Sinkhorn: La couche de sortie utilise une fonction 
    d'activation sigmoid pour restreindre les sorties entre 0 et 1, 
    simulant une probabilité. Si sinkhorn_iter est supérieur à 0, 
    les itérations de Sinkhorn sont appliquées pour rendre 
    la sortie plus proche d'une matrice de permutation.
    """
    def __init__(self, sinkhorn_iter=0):
        super().__init__()
        self.conv_net = SimpleConvNet()  # Crée une instance de SimpleConvNet, un réseau convolutionnel défini ailleurs.
        self.fc1 = nn.Linear(128 * 4, 256)  # Définit une couche linéaire (entièrement connectée) prenant 128 * 4 entrées et produisant 256 sorties.
        self.fc1_bn = nn.BatchNorm1d(256)  # Applique une normalisation par lots à une dimension sur les 256 sorties de la couche précédente.
        self.fc2 = nn.Linear(256, 16)  # Définit une seconde couche linéaire produisant 4 sorties à partir des 256 entrées.
        self.sinkhorn_iter = sinkhorn_iter  # Stocke le nombre d'itérations de Sinkhorn à utiliser lors de la prédiction.
    
    def forward(self, x):
        # Division de l'entrée en 4 morceaux et passage dans le même réseau de neurones convolutionnel.
        x0 = self.conv_net(x[:, :, 0:16, 0:16])
        x1 = self.conv_net(x[:, :, 16:32, 0:16])
        x2 = self.conv_net(x[:, :, 16:32, 16:32])
        x3 = self.conv_net(x[:, :, 0:16, 16:32])

        # Concaténation des caractéristiques de toutes les neuf pièces le long de la dimension 1.
        x = torch.cat([x0, x1, x2, x3], dim=1)
        
        # Traitement par la couche dense.
        x = F.dropout(x, p=0.1, training=self.training)  # Applique un dropout pour réduire le surajustement.
        x = F.relu(self.fc1_bn(self.fc1(x)))  # Applique une couche linéaire, suivie d'une normalisation par lots et d'une activation ReLU.
        x = torch.sigmoid(self.fc2(x))  # Applique une autre couche linéaire et utilise la fonction d'activation sigmoid pour obtenir des probabilités.



        # Si des itérations de Sinkhorn sont spécifiées, les appliquer pour affiner la matrice de sortie.
        if self.sinkhorn_iter > 0:
            x = x.view(-1, 4, 4)  # Remodelage de x en une matrice 3x3 pour chaque élément du batch.
            x = sinkhorn(x, self.sinkhorn_iter)  # Applique les itérations de Sinkhorn pour rendre la matrice plus proche d'une permutation.
            x = x.view(-1, 16)  # Aplatit la matrice en un vecteur de 9 éléments pour la sortie finale.

        return x  # Retourne le vecteur de sortie, qui représente la prédiction de l'arrangement des pièces du puzzle.
