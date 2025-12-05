import numpy as np

def save_network(nn, filename):                     # Fonction de sauvegarde du réseau de neurones.
    data = {}                                       # Dictionnaire pour stocker les poids et biais.
    for i, layer in enumerate(nn.layers):           # Itération sur les couches du réseau.
        data[f"weights_{i}"] = layer.weights        # Stockage des poids.
        data[f"biases_{i}"] = layer.biases          # Stockage des biais.
    np.savez(filename, **data)                      # Sauvegarde dans un fichier .npz.
    
def load_network(nn, filename):                     # Fonction de chargement du réseau de neurones.
    data = np.load(filename)                        # Chargement des données depuis le fichier .npz.
    for i, layer in enumerate(nn.layers):           # Itération sur les couches du réseau.
        layer.weights = data[f"weights_{i}"]        # Restauration des poids.
        layer.biases  = data[f"biases_{i}"]         # Restauration des biais.